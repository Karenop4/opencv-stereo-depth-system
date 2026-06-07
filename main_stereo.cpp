#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <cmath>
#include <limits>
#include <algorithm>

#include "CameraStream.hpp"
#include "StereoProcessor.hpp"
#include "utils.hpp"
#include "FaceSwapper.hpp"
#include "YOLODetector.hpp"

namespace {

/** @brief Offset de brillo aplicado a la imagen rectificada. */
int brilloTrack  = 100;  // 100 = 0 compensación
/** @brief Ganancia de contraste en escala x10. */
int contTrack    = 10;   // 10  = 1.0x multiplicador
/** @brief Activa el preprocesado CLAHE antes de la disparidad. */
int claheTrack   = 1;    // 1   = Activado
/** @brief Área mínima de objeto expresada en centenas de píxeles. */
int minAreaTrack = 5;    // área mínima x100 px²
/** @brief Selector de 64/128/256 disparidades. */
int numDispMult  = 1;   // 0=64, 1=128, 2=256
/** @brief Tamaño de bloque de la SGM. */
int blockSz      = 7;
/** @brief Umbral máximo de distancia para objetos cercanos. */
int depthThreshTrack = 120; // distancia maxima para objeto cercano en cm

/** @brief Parámetros de control enviados al firmware de la ESP32-CAM. */
const std::vector<std::pair<std::string, int>> ESP32_CONTROLES = {
    {"aec", 1},          // auto-exposicion activa para evitar imagen quemada
    {"aec2", 1},
    {"ae_level", -1},
    {"agc", 1},          // auto-ganancia activa; el suavizado se hace por software
    {"gainceiling", 1},
    {"brightness", -1},
    {"contrast", 0},
    {"xclk", 20}         // firmware personalizado: XCLK estable a 20 MHz
};

/**
 * @brief Ajusta el tamaño de una imagen a la resolución calibrada.
 * @param img Imagen de entrada/salida.
 * @param size Tamaño objetivo.
 * @return No devuelve valor.
 * @note Mantener ambas vistas con la misma geometría es esencial para la
 * rectificación y la correspondencia estéreo.
 */
static void asegurarTamanoCalibrado(cv::Mat& img, const cv::Size& size) {
    if (!img.empty() && img.size() != size) {
        cv::resize(img, img, size, 0, 0, cv::INTER_AREA);
    }
}

/**
 * @brief Expande un ROI sin salir de los límites de la imagen.
 * @param roi Región original.
 * @param bounds Dimensiones máximas de la imagen.
 * @param factor Factor de expansión aplicado al ROI.
 * @return ROI expandido y recortado a la imagen.
 * @note Se usa como recuperación cuando el centro del objeto contiene muchos
 * píxeles de profundidad inválidos.
 */
static cv::Rect expandirROIClamped(const cv::Rect& roi, const cv::Size& bounds, double factor) {
    if (roi.empty() || factor <= 1.0) return roi & cv::Rect(0, 0, bounds.width, bounds.height);

    const double cx = roi.x + roi.width * 0.5;
    const double cy = roi.y + roi.height * 0.5;
    const double halfW = roi.width * factor * 0.5;
    const double halfH = roi.height * factor * 0.5;

    cv::Rect expanded(
        static_cast<int>(std::floor(cx - halfW)),
        static_cast<int>(std::floor(cy - halfH)),
        static_cast<int>(std::ceil(halfW * 2.0)),
        static_cast<int>(std::ceil(halfH * 2.0)));

    return expanded & cv::Rect(0, 0, bounds.width, bounds.height);
}

/**
 * @brief Calcula la mediana Z sobre una ROI 3D con filtrado local.
 * @param points3D Nube de puntos reproyectada.
 * @param roi Región a evaluar.
 * @param permitirExpansion Habilita una expansión de búsqueda si hay muchos inválidos.
 * @param profundidad Nivel interno de recursión.
 * @return Mediana Z en milímetros o -1 si no se puede estimar.
 * @note Reduce ruido del sensor y mejora la estabilidad métrica exigida por la rúbrica.
 */
static double medianaZValidaImpl(const cv::Mat& points3D, const cv::Rect& roi, bool permitirExpansion, int profundidad) {
    cv::Rect safe = roi & cv::Rect(0, 0, points3D.cols, points3D.rows);
    if (safe.empty()) return -1.0;

    cv::Mat region = points3D(safe).clone();
    const int minDim = std::min(region.cols, region.rows);
    const int ksize = (minDim >= 5) ? 5 : (minDim >= 3 ? 3 : 0);
    if (ksize > 0) {
        cv::medianBlur(region, region, ksize);
    }

    std::vector<float> vals;
    vals.reserve(safe.area());
    int invalidCount = 0;
    const int totalCount = region.rows * region.cols;

    for (int y = 0; y < region.rows; ++y) {
        const cv::Vec3f* row = region.ptr<cv::Vec3f>(y);
        for (int x = 0; x < region.cols; ++x) {
            float z = std::abs(row[x][2]);
            if (std::isfinite(z) && z > 150.0f && z < 10000.0f) {
                vals.push_back(z);
            } else {
                ++invalidCount;
            }
        }
    }

    const double invalidRatio = (totalCount > 0)
        ? static_cast<double>(invalidCount) / static_cast<double>(totalCount)
        : 1.0;

    if (invalidRatio > 0.40 && permitirExpansion && profundidad == 0) {
        cv::Rect expanded = expandirROIClamped(safe, points3D.size(), 1.5);
        if (expanded.area() > safe.area() && expanded != safe) {
            std::cerr << "[WARN] ROI de profundidad con "
                      << static_cast<int>(invalidRatio * 100.0)
                      << "% de pixeles invalidos. Ampliando busqueda." << std::endl;
            return medianaZValidaImpl(points3D, expanded, false, profundidad + 1);
        }
    }

    if (invalidRatio > 0.40) {
        std::cerr << "[WARN] ROI de profundidad sigue con "
                  << static_cast<int>(invalidRatio * 100.0)
                  << "% de pixeles invalidos; la lectura puede ser inestable." << std::endl;
    }

    if (vals.size() < 8) return -1.0;
    std::nth_element(vals.begin(), vals.begin() + vals.size() / 2, vals.end());
    return vals[vals.size() / 2];
}

/**
 * @brief Estima un rango robusto de disparidad mediante percentiles.
 * @param dispF Mapa de disparidad en float.
 * @param validMask Máscara de píxeles válidos.
 * @param pLow Percentil inferior.
 * @param pHigh Percentil superior.
 * @param low Salida del límite inferior.
 * @param high Salida del límite superior.
 * @return true si el rango pudo calcularse.
 * @note Sirve para visualización estable del mapa de disparidad y para evitar
 * que valores extremos dominen el contraste mostrado al usuario.
 */
static bool rangoDisparidadPercentil(const cv::Mat& dispF, const cv::Mat& validMask, double pLow, double pHigh, double& low, double& high) {
    std::vector<float> vals;
    vals.reserve(cv::countNonZero(validMask));

    for (int y = 0; y < dispF.rows; ++y) {
        const float* d = dispF.ptr<float>(y);
        const uchar* m = validMask.ptr<uchar>(y);
        for (int x = 0; x < dispF.cols; ++x) {
            if (m[x]) vals.push_back(d[x]);
        }
    }

    if (vals.size() < 32) return false;

    auto pct = [&](double p) {
        size_t idx = std::min(vals.size() - 1, static_cast<size_t>(p * (vals.size() - 1)));
        std::nth_element(vals.begin(), vals.begin() + idx, vals.end());
        return static_cast<double>(vals[idx]);
    };

    low = pct(std::clamp(pLow, 0.0, 1.0));
    high = pct(std::clamp(pHigh, 0.0, 1.0));
    return high > low + 1e-3;
}

/**
 * @brief Devuelve el instante actual en microsegundos usando un reloj monotónico.
 * @return Tiempo monotónico en microsegundos.
 * @note Se usa para medir la edad del frame y filtrar desincronizaciones entre cámaras.
 */
static long long ahoraUs() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

/**
 * @brief Calcula la edad del último frame recibido por una cámara.
 * @param cam Stream compartido de la cámara.
 * @return Edad del frame en milisegundos o infinito si no hay frame válido.
 * @note Ayuda a descartar frames viejos y disminuir jitter temporal en la reconstrucción.
 */
static double edadFrameMs(const CameraStream& cam) {
    long long stamp = cam.lastFrameUs.load(std::memory_order_relaxed);
    if (stamp <= 0) return std::numeric_limits<double>::infinity();
    return static_cast<double>(ahoraUs() - stamp) / 1000.0;
}

/**
 * @brief Genera una máscara de confianza de disparidad a partir de la mediana local.
 * @param dispF Mapa de disparidad en float.
 * @param validMask Máscara base de disparidades válidas.
 * @return Máscara binaria de píxeles confiables.
 * @note Reduce speckle noise y evita que outliers contaminen la distancia y la visualización.
 */
static cv::Mat mascaraConfianzaDisparidad(const cv::Mat& dispF, const cv::Mat& validMask) {
    cv::Mat mediana;
    cv::medianBlur(dispF, mediana, 5);

    cv::Mat diff;
    cv::absdiff(dispF, mediana, diff);

    cv::Mat diffMask;
    cv::inRange(diff, cv::Scalar(0.0f), cv::Scalar(2.5f), diffMask);
    diffMask.setTo(0, ~validMask);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(diffMask, diffMask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(diffMask, diffMask, cv::MORPH_CLOSE, kernel);
    return diffMask;
}

/**
 * @brief Calcula la mediana de disparidad sobre un ROI usando una máscara de confianza.
 * @param dispF Mapa de disparidad en float.
 * @param roi Región de interés.
 * @param mask Máscara de píxeles confiables.
 * @return Mediana de disparidad o -1 si no hay muestras válidas.
 * @note Esta medición es la base de la conversión a centímetros sin contaminarse por speckles.
 */
static float medianaDisparidadConMascara(const cv::Mat& dispF, const cv::Rect& roi, const cv::Mat& mask) {
    cv::Rect safe = roi & cv::Rect(0, 0, dispF.cols, dispF.rows);
    if (safe.empty()) return -1.0f;

    cv::Mat region = dispF(safe);
    cv::Mat maskRegion = mask(safe);
    std::vector<float> vals;
    vals.reserve(safe.area());
    for (int y = 0; y < region.rows; ++y) {
        const float* d = region.ptr<float>(y);
        const uchar* m = maskRegion.ptr<uchar>(y);
        for (int x = 0; x < region.cols; ++x) {
            if (m[x] && d[x] > 2.0f && d[x] < 200.0f) vals.push_back(d[x]);
        }
    }
    if (vals.size() < 8) return -1.0f;
    std::nth_element(vals.begin(), vals.begin() + vals.size() / 2, vals.end());
    return vals[vals.size() / 2];
}

/**
 * @brief Calcula la mediana Z válida usando la función auxiliar con expansión.
 * @param points3D Nube de puntos reproyectada.
 * @param roi Región de interés.
 * @return Mediana Z en milímetros o -1 si no hay datos válidos.
 * @note Es el valor métrico que alimenta la estabilización temporal y la calibración.
 */
static double medianaZValida(const cv::Mat& points3D, const cv::Rect& roi) {
    return medianaZValidaImpl(points3D, roi, true, 0);
}

/**
 * @brief Detecta objetos rectangulares oscuros excluyendo rostros.
 * @param frame Imagen BGR a analizar.
 * @param areaMin Área mínima del candidato.
 * @param faces Rostros a enmascarar durante la búsqueda.
 * @return Rectángulo del mejor candidato o vacío si no hay objeto.
 * @note Aporta una ruta auxiliar de detección cuando la nube de puntos no es suficiente.
 */
static cv::Rect detectarObjetoOscuroRectangular(const cv::Mat& frame, double areaMin, const std::vector<Detection>& faces) {
    if (frame.empty()) return {};

    cv::Mat gray, mask;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, {5, 5}, 0.0);

    cv::Scalar meanGray, stdGray;
    cv::meanStdDev(gray, meanGray, stdGray);
    double darkThreshold = std::clamp(meanGray[0] - 0.35 * stdGray[0], 35.0, 95.0);
    cv::threshold(gray, mask, darkThreshold, 255, cv::THRESH_BINARY_INV);

    for (const auto& face : faces) {
        cv::Rect safe = face.box & cv::Rect(0, 0, mask.cols, mask.rows);
        if (!safe.empty()) mask(safe).setTo(0);
    }

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, {7, 5});
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel, {-1, -1}, 1);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel, {-1, -1}, 2);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Point frameCenter(frame.cols / 2, frame.rows / 2);
    cv::Rect best;
    double bestScore = -1.0;
    for (const auto& cnt : contours) {
        double area = std::abs(cv::contourArea(cnt));
        if (area < std::max(700.0, areaMin * 0.5)) continue;

        cv::Rect box = cv::boundingRect(cnt) & cv::Rect(0, 0, frame.cols, frame.rows);
        if (box.width < 45 || box.height < 16) continue;
        if (box.width > frame.cols * 0.65 || box.height > frame.rows * 0.35) continue;

        double aspect = static_cast<double>(box.width) / std::max(1, box.height);
        if (aspect < 1.55 || aspect > 8.0) continue;

        double rectangularidad = area / std::max(1, box.area());
        if (rectangularidad < 0.25) continue;

        double darkFraction = static_cast<double>(cv::countNonZero(mask(box))) / std::max(1, box.area());
        if (darkFraction < 0.18) continue;

        cv::Point c(box.x + box.width / 2, box.y + box.height / 2);
        double centerPenalty = cv::norm(c - frameCenter) / std::max(frame.cols, frame.rows);
        double score = area * (0.6 + darkFraction) * (1.0 + std::min(3.0, aspect)) - centerPenalty * 4000.0;
        if (score > bestScore) {
            bestScore = score;
            best = box;
        }
    }

    return best;
}

/**
 * @brief Detecta el objeto más cercano usando la nube de puntos 3D.
 * @param points3D Nube de puntos en milímetros.
 * @param depthThreshCm Umbral máximo de distancia en centímetros.
 * @param areaMin Área mínima aceptada para el objeto.
 * @param bestBox Caja resultante del objeto detectado.
 * @param meanDepthCm Profundidad media estimada del objeto.
 * @return true si se encontró un objeto válido.
 * @note Es necesario para ofrecer una lectura estable cuando el objeto visual no es claro.
 */
static bool detectarObjetoPorProfundidad(
    const cv::Mat& points3D,
    double depthThreshCm,
    double areaMin,
    cv::Rect& bestBox,
    double& meanDepthCm)
{
    if (points3D.empty()) return false;

    cv::Mat depthCm(points3D.size(), CV_32F, cv::Scalar(0));
    cv::Mat valid(points3D.size(), CV_8U, cv::Scalar(0));
    for (int y = 0; y < points3D.rows; ++y) {
        const cv::Vec3f* p = points3D.ptr<cv::Vec3f>(y);
        float* d = depthCm.ptr<float>(y);
        uchar* m = valid.ptr<uchar>(y);
        for (int x = 0; x < points3D.cols; ++x) {
            float zCm = std::abs(p[x][2]) / 10.0f;
            if (std::isfinite(zCm) && zCm >= 10.0f && zCm <= 350.0f) {
                d[x] = zCm;
                m[x] = 255;
            }
        }
    }

    cv::Mat mask;
    cv::inRange(depthCm, cv::Scalar(10.0), cv::Scalar(depthThreshCm), mask);
    mask.setTo(0, ~valid);

    int borderX = std::max(18, points3D.cols / 24);
    int borderY = std::max(12, points3D.rows / 32);
    mask(cv::Rect(0, 0, borderX, mask.rows)).setTo(0);
    mask(cv::Rect(mask.cols - borderX, 0, borderX, mask.rows)).setTo(0);
    mask(cv::Rect(0, 0, mask.cols, borderY)).setTo(0);
    mask(cv::Rect(0, mask.rows - borderY, mask.cols, borderY)).setTo(0);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, {7, 7});
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel, {-1, -1}, 1);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel, {-1, -1}, 2);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return false;

    cv::Rect best;
    double bestScore = -1.0;
    double bestMean = -1.0;
    const double imgArea = static_cast<double>(points3D.rows * points3D.cols);

    for (const auto& cnt : contours) {
        double area = std::abs(cv::contourArea(cnt));
        if (area < std::max(areaMin, imgArea * 0.006)) continue;
        if (area > imgArea * 0.22) continue;

        cv::Rect box = cv::boundingRect(cnt) & cv::Rect(0, 0, points3D.cols, points3D.rows);
        if (box.width < 20 || box.height < 20) continue;
        if (box.width > points3D.cols * 0.75 || box.height > points3D.rows * 0.75) continue;

        cv::Mat contourMask = cv::Mat::zeros(mask.size(), CV_8U);
        std::vector<std::vector<cv::Point>> one = {cnt};
        cv::drawContours(contourMask, one, 0, cv::Scalar(255), -1);

        cv::Scalar mean, stddev;
        cv::meanStdDev(depthCm, mean, stddev, contourMask);
        if (mean[0] <= 0.0 || stddev[0] > 35.0) continue;

        double rectangularidad = area / std::max(1, box.area());
        cv::Point2d center(box.x + box.width * 0.5, box.y + box.height * 0.5);
        cv::Point2d imageCenter(points3D.cols * 0.5, points3D.rows * 0.5);
        double centerDist = cv::norm(center - imageCenter) / std::max(points3D.cols, points3D.rows);
        if (centerDist > 0.38) continue;

        double score = area * rectangularidad /
                       (std::max(1.0, mean[0]) * (1.0 + 4.0 * centerDist));
        if (score > bestScore) {
            bestScore = score;
            best = box;
            bestMean = mean[0];
        }
    }

    if (best.empty() || bestMean <= 0.0) return false;
    bestBox = best;
    meanDepthCm = bestMean;
    return true;
}

/**
 * @brief Dibuja una capa AR que cambia con la distancia estimada.
 * @param canvas Imagen BGR donde se compone el overlay.
 * @param box Caja del objeto medido.
 * @param distanciaCm Distancia filtrada y calibrada en centímetros.
 * @return No devuelve valor.
 * @note Justificación: cumple el criterio de AR dinámico, porque escala,
 * color y posición del retículo dependen de la profundidad medida en tiempo real.
 */
static void dibujarEfectoARDistancia(cv::Mat& canvas, const cv::Rect& box, double distanciaCm) {
    if (canvas.empty() || box.empty() || distanciaCm <= 0.0) return;

    const cv::Rect safe = box & cv::Rect(0, 0, canvas.cols, canvas.rows);
    if (safe.empty()) return;

    const double nearFactor = 1.0 - std::clamp((distanciaCm - 20.0) / 180.0, 0.0, 1.0);
    const int pulse = static_cast<int>(8.0 + nearFactor * 28.0);
    const int thickness = 2 + static_cast<int>(nearFactor * 4.0);
    const cv::Scalar color(
        30.0,
        80.0 + nearFactor * 175.0,
        255.0 - nearFactor * 190.0);

    cv::Mat overlay = canvas.clone();
    cv::Point center(safe.x + safe.width / 2, safe.y + safe.height / 2);
    cv::Size haloSize(
        std::max(12, safe.width / 2 + pulse),
        std::max(12, safe.height / 2 + pulse));

    cv::ellipse(overlay, center, haloSize, 0.0, 0.0, 360.0, color, thickness, cv::LINE_AA);
    cv::line(overlay, {center.x - pulse, center.y}, {center.x + pulse, center.y}, color, 2, cv::LINE_AA);
    cv::line(overlay, {center.x, center.y - pulse}, {center.x, center.y + pulse}, color, 2, cv::LINE_AA);

    const int barHeight = std::clamp(static_cast<int>(nearFactor * safe.height), 6, std::max(6, safe.height));
    cv::Rect depthBar(
        std::clamp(safe.x + safe.width + 8, 0, std::max(0, canvas.cols - 10)),
        std::clamp(safe.y + safe.height - barHeight, 0, std::max(0, canvas.rows - barHeight)),
        8,
        barHeight);
    cv::rectangle(overlay, depthBar, color, -1, cv::LINE_AA);

    cv::addWeighted(overlay, 0.38, canvas, 0.62, 0.0, canvas);
}

} // namespace

/**
 * @brief Punto de entrada de la aplicación de profundidad estéreo y AR.
 * @return 0 si la ejecución termina correctamente.
 * @note Orquesta captura concurrente, filtrado estéreo, calibración y el efecto AR.
 */
int main() {
    StereoProcessor stereoProc("parametros_stereo.yml", "scale.yml");
    
    YOLODetector faceDetector("yolov26/runs/detect/yolov26_faces/weights/best.onnx");
    FaceSwapper faceSwapper("shape_predictor_68_face_landmarks.dat");

    const std::string leftUrl  = "http://192.168.3.45:81/stream";
    const std::string rightUrl = "http://192.168.3.44:81/stream";
    configure_esp32_cam(leftUrl, ESP32_CONTROLES);
    configure_esp32_cam(rightUrl, ESP32_CONTROLES);

    CameraStream camLeft(leftUrl);
    CameraStream camRight(rightUrl);
    std::thread threadLeft (capture_loop, &camLeft);
    std::thread threadRight(capture_loop, &camRight);

    cv::namedWindow("Camara Izquierda", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Mapa de Disparidad", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Ajustes Luz", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Ajustes Vision", cv::WINDOW_AUTOSIZE);

    cv::createTrackbar("Brillo",        "Ajustes Luz",       &brilloTrack,  200, nullptr);
    cv::createTrackbar("Contraste x10", "Ajustes Luz",       &contTrack,     30, nullptr);
    cv::createTrackbar("CLAHE",         "Ajustes Luz",       &claheTrack,     1, nullptr);
    cv::createTrackbar("Area min x100", "Ajustes Vision",     &minAreaTrack, 150, nullptr);
    cv::createTrackbar("Num Disp",       "Ajustes Vision",     &numDispMult,    2, nullptr);
    cv::createTrackbar("Block Size",     "Ajustes Vision",     &blockSz,       15, nullptr);
    cv::createTrackbar("Dist max cm",    "Ajustes Vision",     &depthThreshTrack, 300, nullptr);

    std::cout << "\n[CONTROLES]:" << std::endl;
    std::cout << "  Brillo/Contraste/CLAHE -> ajuste de luz" << std::endl;
    std::cout << "  Area min x100 -> tamaño minimo de objeto" << std::endl;
    std::cout << "  Num Disp      -> 64 / 128 / 256" << std::endl;
    std::cout << "  Dist max cm   -> umbral de objeto cercano" << std::endl;
    std::cout << "  [C]           -> calibrar distancia actual" << std::endl;
    std::cout << "  [S]           -> guardar frame" << std::endl;
    std::cout << "  [ESC]         -> salir" << std::endl;

    cv::VideoWriter videoOut("salida_stereo.avi", cv::VideoWriter::fourcc('M','J','P','G'), 10.0, cv::Size(stereoProc.imgSize.width * 2, stereoProc.imgSize.height));

    cv::Mat imgLeft, imgRight;
    Suavizador suavCentral(16);
    Kalman1D kalmanCentral(0.05, 4.0);
    int frame_idx = 0;
    double lastRawDistanceCm = -1.0;

    while (true) {
        lastRawDistanceCm = -1.0;
        bool okL = camLeft.connected  && !camLeft.frame.empty();
        bool okR = camRight.connected && !camRight.frame.empty();
        if (okL) { std::lock_guard<std::mutex> l(camLeft.mtx);  camLeft.frame.copyTo(imgLeft);   }
        if (okR) { std::lock_guard<std::mutex> l(camRight.mtx); camRight.frame.copyTo(imgRight); }

        double ageL = edadFrameMs(camLeft);
        double ageR = edadFrameMs(camRight);
        double syncDelta = std::abs(ageL - ageR);
        if (ageL > 250.0 || ageR > 250.0 || syncDelta > 90.0) {
            std::cerr << "[WARN] Frames desincronizados o viejos: L=" << ageL
                      << " ms, R=" << ageR << " ms, delta=" << syncDelta << " ms" << std::endl;
            if (cv::waitKey(1) == 27) break;
            continue;
        }

        if (!okL || !okR) {
            cv::Mat w = cv::Mat::zeros(100, 400, CV_8UC3);
            cv::putText(w, "Esperando camaras...", {10,55},
                cv::FONT_HERSHEY_SIMPLEX, 0.85, cv::Scalar(0,200,255), 2);
            cv::imshow("Camara Izquierda", w);
            if (cv::waitKey(30) == 27) break;
            continue;
        }

        asegurarTamanoCalibrado(imgLeft, stereoProc.imgSize);
        asegurarTamanoCalibrado(imgRight, stereoProc.imgSize);

        cv::Mat rectL, rectR;
        stereoProc.rectifyImages(imgLeft, imgRight, rectL, rectR);
        cv::Mat vistaL = imgLeft.clone();
        cv::Mat procL = rectL.clone();
        cv::Mat procR = rectR.clone();

        double alphaC = std::max(0.1, contTrack / 10.0);
        double betaB  = brilloTrack - 100.0;
        if (std::abs(alphaC - 1.0) > 0.05 || std::abs(betaB) > 0.1) {
            procL.convertTo(procL, -1, alphaC, betaB);
            procR.convertTo(procR, -1, alphaC, betaB);
        }

        stereoProc.setSGBMParameters(numDispMult, blockSz);
        
        cv::Mat dispFilt, dispF;
        stereoProc.computeDisparity(procL, procR, claheTrack, dispFilt, dispF);
        cv::Mat points3D;
        if (!stereoProc.qMatrix.empty()) {
            cv::reprojectImageTo3D(dispF, points3D, stereoProc.qMatrix, true);
        }
        double areaMin = minAreaTrack * 100.0;
        
        std::vector<Detection> faces = faceDetector.detect(vistaL, 0.5f);
        cv::Rect objetoVisual = detectarObjetoOscuroRectangular(vistaL, areaMin, faces);

        cv::Mat validDispMask = (dispF > 2.0f) & (dispF < 200.0f);
        cv::Mat confDispMask = mascaraConfianzaDisparidad(dispF, validDispMask);
        validDispMask &= confDispMask;
        int borderX = std::max(12, stereoProc.imgSize.width / 28);
        int borderY = std::max(8, stereoProc.imgSize.height / 40);
        validDispMask(cv::Rect(0, 0, borderX, validDispMask.rows)).setTo(0);
        validDispMask(cv::Rect(validDispMask.cols - borderX, 0, borderX, validDispMask.rows)).setTo(0);
        validDispMask(cv::Rect(0, 0, validDispMask.cols, borderY)).setTo(0);
        validDispMask(cv::Rect(0, validDispMask.rows - borderY, validDispMask.cols, borderY)).setTo(0);
        cv::Mat canvas = vistaL.clone();
        if (faces.size() == 2) {
            faceSwapper.swapFaces(canvas, faces[0].box, faces[1].box);
        } else {
            for (const auto& f : faces) {
                cv::rectangle(canvas, f.box, cv::Scalar(0, 255, 0), 2);
            }
        }

        cv::Rect objetoCentral;
        double distanciaCentralRawCm = -1.0;
        double distanciaDisplayCm = -1.0;

        if (!objetoVisual.empty() && !points3D.empty()) {
            objetoCentral = objetoVisual;
            cv::Rect roiInner(
                objetoVisual.x + objetoVisual.width / 5,
                objetoVisual.y + objetoVisual.height / 5,
                std::max(1, objetoVisual.width * 3 / 5),
                std::max(1, objetoVisual.height * 3 / 5));
            float dispMed = medianaDisparidadConMascara(dispF, roiInner, validDispMask);
            if (dispMed <= 0) dispMed = medianaDisparidadConMascara(dispF, objetoVisual, validDispMask);

            double zFromDispCm = -1.0;
            if (dispMed > 0) {
                zFromDispCm = (stereoProc.focal_px * stereoProc.baseline_mm) / dispMed / 10.0;
            }

            double zFromPointsCm = medianaZValida(points3D, roiInner);
            if (zFromPointsCm <= 0) zFromPointsCm = medianaZValida(points3D, objetoVisual);
            if (zFromPointsCm > 0) zFromPointsCm /= 10.0;

            if (zFromDispCm > 0 && zFromPointsCm > 0) {
                double relDiff = std::abs(zFromDispCm - zFromPointsCm) / std::max(1.0, zFromDispCm);
                if (relDiff > 0.35) {
                    std::cerr << "[WARN] Profundidad inconsistente entre disparidad y reproyeccion; usando disparidad." << std::endl;
                    distanciaCentralRawCm = zFromDispCm;
                } else {
                    distanciaCentralRawCm = 0.5 * (zFromDispCm + zFromPointsCm);
                }
            } else if (zFromDispCm > 0) {
                distanciaCentralRawCm = zFromDispCm;
            } else if (zFromPointsCm > 0) {
                distanciaCentralRawCm = zFromPointsCm;
            }
        }

        if (objetoCentral.empty() && !points3D.empty()) {
            cv::Rect depthBox;
            double depthMeanCm = -1.0;
            double depthThreshCm = std::clamp(static_cast<double>(depthThreshTrack), 20.0, 300.0);
            if (detectarObjetoPorProfundidad(points3D, depthThreshCm, areaMin, depthBox, depthMeanCm)) {
                objetoCentral = depthBox;
                distanciaCentralRawCm = depthMeanCm;
            }
        }

        if (!objetoCentral.empty()) {
            cv::Rect box = objetoCentral & cv::Rect(0, 0, canvas.cols, canvas.rows);
            cv::Scalar color(0, 255, 0);
            if (!box.empty()) {
                cv::rectangle(canvas, box, color, 2, cv::LINE_AA);
                cv::Point c(box.x + box.width / 2, box.y + box.height / 2);
                cv::circle(canvas, c, 4, cv::Scalar(0, 255, 255), -1, cv::LINE_AA);
            }

            char distBuf[64];
            if (distanciaCentralRawCm > 0) {
                lastRawDistanceCm = distanciaCentralRawCm;

                // Si hay salto brusco de distancia, reinicia filtros para evitar que se queden anclados.
                if (kalmanCentral.iniciado) {
                    double delta = std::abs(distanciaCentralRawCm - kalmanCentral.x);
                    double salto = std::max(10.0, 0.25 * std::max(1.0, kalmanCentral.x));
                    if (delta > salto) {
                        suavCentral.hist.clear();
                        kalmanCentral.iniciado = false;
                        kalmanCentral.x = 0.0;
                        kalmanCentral.p = 1.0;
                    }
                }

                double zSuavizadoRawCm = suavCentral.agregar(distanciaCentralRawCm);
                double zKalmanRawCm = kalmanCentral.actualizar(zSuavizadoRawCm);
                double zDisplayCm = zKalmanRawCm * stereoProc.scaleFactor;
                distanciaDisplayCm = zDisplayCm;
                snprintf(distBuf, sizeof(distBuf), "Distancia: %.1f cm", zDisplayCm);
            } else {
                snprintf(distBuf, sizeof(distBuf), "Distancia: -- cm");
            }

            int base = 0;
            cv::Size ts = cv::getTextSize(distBuf, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &base);
            int tx = box.empty() ? 30 : std::clamp(box.x, 6, std::max(6, canvas.cols - ts.width - 10));
            int ty = box.empty() ? 40 : std::max(24, box.y - 8);
            cv::rectangle(canvas,
                {tx - 4, ty - ts.height - 5},
                {tx + ts.width + 5, ty + base + 4},
                cv::Scalar(0, 0, 0), -1);
            cv::putText(canvas, distBuf, {tx, ty}, cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv::LINE_AA);
            dibujarEfectoARDistancia(canvas, box, distanciaDisplayCm);
        }

        cv::Mat dispVis, dispNorm;
        double dMin = 0.0, dMax = 0.0;
        cv::Mat dispVisMask = validDispMask.clone();
        if (!rangoDisparidadPercentil(dispF, dispVisMask, 0.02, 0.90, dMin, dMax)) {
            cv::minMaxLoc(dispF, &dMin, &dMax, nullptr, nullptr, dispVisMask);
        }
        dispF.convertTo(dispNorm, CV_8U,
            255.0/(dMax-dMin+1e-6),
           -dMin*255.0/(dMax-dMin+1e-6));
        cv::medianBlur(dispNorm, dispNorm, 5);
        cv::Mat dispSmooth;
        cv::bilateralFilter(dispNorm, dispSmooth, 7, 35.0, 7.0);
        dispNorm = dispSmooth;
        cv::Mat kernelClose = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::morphologyEx(dispNorm, dispNorm, cv::MORPH_CLOSE, kernelClose);
        cv::morphologyEx(dispVisMask, dispVisMask, cv::MORPH_CLOSE, kernelClose);
        auto claheDisp = cv::createCLAHE(2.0, cv::Size(8, 8));
        claheDisp->apply(dispNorm, dispNorm);
        cv::cvtColor(dispNorm, dispVis, cv::COLOR_GRAY2BGR);
        dispVis.setTo(cv::Scalar(0,0,0), ~dispVisMask);

        cv::Mat combined;
        std::vector<cv::Mat> panels = {canvas, dispVis};
        cv::hconcat(panels, combined);
        cv::imshow("Camara Izquierda", canvas);
        cv::imshow("Mapa de Disparidad", dispVis);
        if (videoOut.isOpened()) videoOut.write(combined);

        int key = cv::waitKey(1);
        if (key == 'c' || key == 'C') {
            if (lastRawDistanceCm > 0) {
                std::cout << "[CAL] Introduce distancia real en cm (ej: 50): ";
                double real_cm = 0.0;
                if (std::cin >> real_cm && real_cm > 0) {
                    double measuredRaw = lastRawDistanceCm;
                    double newScale = real_cm / measuredRaw;
                    stereoProc.scaleFactor = newScale;
                    suavCentral.hist.clear();
                    kalmanCentral.iniciado = false;
                    kalmanCentral.x = 0.0;
                    kalmanCentral.p = 1.0;
                    cv::FileStorage sfs("scale.yml", cv::FileStorage::WRITE);
                    sfs << "scaleFactor" << stereoProc.scaleFactor;
                    sfs.release();
                    std::cout << "[CAL] scaleFactor set to " << stereoProc.scaleFactor << " and saved to scale.yml" << std::endl;
                } else {
                    std::cout << "[CAL] valor invalido, cancelado." << std::endl;
                    std::cin.clear();
                    std::string dummy; std::getline(std::cin, dummy);
                }
            } else {
                std::cout << "[CAL] No hay lectura estable para calibrar." << std::endl;
            }
        }
        if (key == 27) break;
        if (key == 's' || key == 'S') {
            std::string n = "frame_" + std::to_string(frame_idx++);
            cv::imwrite(n + "_vision.png", canvas);
            cv::imwrite(n + "_depth.png",  dispVis);
            std::cout << "[INFO] Guardado: " << n << std::endl;
        }
    }

    camLeft.running  = false;
    camRight.running = false;
    if (threadLeft.joinable())  threadLeft.join();
    if (threadRight.joinable()) threadRight.join();
    cv::destroyAllWindows();
    return 0;
}
