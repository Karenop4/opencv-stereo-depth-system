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
#include "DepthUtils.hpp"
#include "utils.hpp"
#include "FaceSwapper.hpp"
#include "YOLODetector.hpp"

namespace {

using depth::cajaDesdeMascara;
using depth::detectarObjetoPorProfundidad;
using depth::distanciaRobustaCm;
using depth::elegirModoDisparidadAuto;
using depth::filtrarComponentesPequenos;
using depth::mascaraConfianzaDisparidad;
using depth::mascaraPrimerPlanoCentral;
using depth::medianaDisparidadConMascara;
using depth::medicionDistanciaPlausible;
using depth::percentilDisparidadConMascara;
using depth::rangoDisparidadPercentil;
using depth::roiCentralMedicion;

/** @brief Offset de brillo aplicado a la imagen rectificada. */
int brilloTrack  = 100;  // 100 = 0 compensación
/** @brief Ganancia de contraste en escala x10. */
int contTrack    = 10;   // 10  = 1.0x multiplicador
/** @brief Activa el preprocesado CLAHE antes de la disparidad. */
int claheTrack   = 0;    // 1   = Activado
/** @brief Activa balance de blancos por software para compensar tinte verde. */
int wbSoftTrack  = 1;    // 1   = Activado
/** @brief Área mínima de objeto expresada en centenas de píxeles. */
int minAreaTrack = 5;    // área mínima x100 px²
/** @brief Selector automatico de 64/128/192 disparidades. */
int numDispAuto  = 1;   // 128 cubre ~33 cm y evita el recorte fuerte de 192
/** @brief Tamaño de bloque de la SGM. */
int blockSz      = 11;
/** @brief Umbral máximo de distancia para objetos cercanos. */
int depthThreshTrack = 120; // distancia maxima para objeto cercano en cm
/** @brief Activa YOLO/FaceSwap; apagarlo ayuda a calibrar profundidad sin lag. */
int arYoloTrack = 0;
/** @brief Activa grabación de video; apagada por defecto para evitar lag. */
int recordTrack = 0;
/** @brief Prioriza la medición cercana que usa solo el componente central. */
int modoCercaTrack = 1;

/** @brief Parámetros de control enviados al firmware de la ESP32-CAM. */
const std::vector<std::pair<std::string, int>> ESP32_CONTROLES = {
    {"aec", 0},          // AEC desactivado: paridad luminica entre sensores
    {"aec2", 0},
    {"aec_value", 450},  // exposicion manual inicial; ajustar si la imagen queda oscura
    {"ae_level", 0},
    {"agc", 0},          // AGC desactivado: evita saltos de ganancia por camara
    {"agc_gain", 8},     // ganancia manual moderada para bajar ruido respecto a AGC
    {"awb", 0},          // AWB desactivado: color/iluminacion estables
    {"awb_gain", 0},
    {"wb_mode", 0},
    {"hmirror", 0},
    {"vflip", 0},
    {"gainceiling", 0},
    {"brightness", 0},
    {"contrast", 0},
    {"saturation", 0},
    {"bpc", 1},
    {"wpc", 1},
    {"raw_gma", 1},
    {"lenc", 1},
    {"dcw", 1},
    {"xclk", 20}         // uso normal: 20 MHz; documentar prueba a 6 MHz en firmware/informe
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
 * @brief Corrige dominante de color con balance gray-world acotado.
 * @param src Imagen BGR de entrada.
 * @param dst Imagen BGR corregida.
 * @return No devuelve valor.
 * @note Compensa el tinte verde producido al bloquear AWB en firmware, sin
 * reactivar automatismos del sensor que romperían la paridad temporal.
 */
static void balanceBlancosGrayWorld(const cv::Mat& src, cv::Mat& dst) {
    if (src.empty() || src.channels() != 3) {
        src.copyTo(dst);
        return;
    }

    cv::Scalar meanBgr = cv::mean(src);
    double gray = (meanBgr[0] + meanBgr[1] + meanBgr[2]) / 3.0;
    std::vector<cv::Mat> channels;
    cv::split(src, channels);
    for (int i = 0; i < 3; ++i) {
        double gain = gray / std::max(1.0, meanBgr[i]);
        gain = std::clamp(gain, 0.65, 1.55);
        channels[i].convertTo(channels[i], -1, gain, 0.0);
    }
    cv::merge(channels, dst);
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
 * @brief Filtra y ordena detecciones faciales para el efecto AR.
 * @param detections Detecciones crudas del YOLO.
 * @param bounds Tamaño del frame donde se aplicará el FaceSwap.
 * @return Rostros válidos, recortados al frame y ordenados por confianza.
 * @note Evita que cajas parcialmente fuera de imagen rompan el predictor de landmarks.
 */
static std::vector<Detection> prepararRostrosParaAR(const std::vector<Detection>& detections, const cv::Size& bounds) {
    std::vector<Detection> faces;
    const cv::Rect frameBounds(0, 0, bounds.width, bounds.height);

    for (Detection det : detections) {
        det.box &= frameBounds;
        if (det.box.width < 24 || det.box.height < 24) continue;
        if (det.box.area() < 900) continue;
        faces.push_back(det);
    }

    std::sort(faces.begin(), faces.end(), [](const Detection& a, const Detection& b) {
        return a.conf > b.conf;
    });
    return faces;
}

/**
 * @brief Elige dos rostros estables para intercambiar.
 * @param faces Detecciones faciales ya filtradas.
 * @return Par de cajas en orden izquierda-derecha, o vacío si no hay dos rostros.
 * @note Prioriza rostros grandes entre los más confiables para estabilizar landmarks.
 */
static std::vector<cv::Rect> seleccionarRostrosSwap(const std::vector<Detection>& faces) {
    if (faces.size() < 2) return {};

    std::vector<Detection> candidates = faces;
    if (candidates.size() > 4) candidates.resize(4);

    std::sort(candidates.begin(), candidates.end(), [](const Detection& a, const Detection& b) {
        return a.box.area() > b.box.area();
    });

    std::vector<cv::Rect> boxes = {candidates[0].box, candidates[1].box};
    std::sort(boxes.begin(), boxes.end(), [](const cv::Rect& a, const cv::Rect& b) {
        return a.x + a.width * 0.5 < b.x + b.width * 0.5;
    });
    return boxes;
}


} // namespace

/**
 * @brief Punto de entrada de la aplicación de profundidad estéreo y AR.
 * @return 0 si la ejecución termina correctamente.
 * @note Orquesta captura concurrente, filtrado estéreo, calibración y el efecto AR.
 */
int main(int argc, char** argv) {
    // Carga calibracion estereo, mapas de rectificacion y factor de escala.
    StereoProcessor stereoProc("parametros_stereo.yml", "scale.yml");
    // Inicializa los dos modulos "pesados" del efecto AR: detector facial y face swap.
    YOLODetector faceDetector("yolov26/runs/detect/yolov26_faces/weights/best.onnx");
    FaceSwapper faceSwapper("shape_predictor_68_face_landmarks.dat");

    // URLs por defecto de ambas ESP32-CAM; se pueden sobreescribir por argumentos.
    std::string leftUrl  = "http://192.168.3.45:81/stream";
    std::string rightUrl = "http://192.168.3.44:81/stream";
    if (argc >= 3) {
        leftUrl = argv[1];
        rightUrl = argv[2];
    } else {
        std::cout << "[INFO] Uso: ./main_stereo <url_izquierda> <url_derecha>" << std::endl;
        std::cout << "[INFO] Usando URLs por defecto del codigo." << std::endl;
    }
    std::cout << "[INFO] Camara izquierda: " << leftUrl << std::endl;
    std::cout << "[INFO] Camara derecha:   " << rightUrl << std::endl;

    // Aplica controles de firmware para fijar exposicion, ganancia y color antes de medir profundidad.
    configure_esp32_cam(leftUrl, ESP32_CONTROLES);
    configure_esp32_cam(rightUrl, ESP32_CONTROLES);

    // Crea el estado compartido de cada camara y levanta un hilo de captura por stream.
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
    cv::createTrackbar("WB software",   "Ajustes Luz",       &wbSoftTrack,    1, nullptr);
    cv::createTrackbar("Area min x100", "Ajustes Vision",     &minAreaTrack, 150, nullptr);
    cv::createTrackbar("Block Size",     "Ajustes Vision",     &blockSz,       15, nullptr);
    cv::createTrackbar("Dist max cm",    "Ajustes Vision",     &depthThreshTrack, 300, nullptr);
    cv::createTrackbar("AR YOLO",        "Ajustes Vision",     &arYoloTrack,    1, nullptr);
    cv::createTrackbar("Grabar",         "Ajustes Vision",     &recordTrack,    1, nullptr);
    cv::createTrackbar("Modo cerca",     "Ajustes Vision",     &modoCercaTrack, 1, nullptr);

    std::cout << "\n[CONTROLES]:" << std::endl;
    std::cout << "  Brillo/Contraste/CLAHE -> ajuste de luz" << std::endl;
    std::cout << "  Area min x100 -> tamaño minimo de objeto" << std::endl;
    std::cout << "  Num Disp      -> automatico 64 / 128 / 192" << std::endl;
    std::cout << "  Dist max cm   -> umbral de objeto cercano" << std::endl;
    std::cout << "  FaceSwap      -> automatico con 2 rostros YOLO" << std::endl;
    std::cout << "  [C]           -> calibrar distancia actual" << std::endl;
    std::cout << "  [S]           -> guardar frame" << std::endl;
    std::cout << "  [ESC]         -> salir" << std::endl;

    // El video grabado junta imagen izquierda procesada + mapa de disparidad mostrado.
    cv::VideoWriter videoOut("salida_stereo.avi", cv::VideoWriter::fourcc('M','J','P','G'), 10.0, cv::Size(stereoProc.imgSize.width * 2, stereoProc.imgSize.height));

    cv::Mat imgLeft, imgRight;
    // Media movil y Kalman estabilizan la Z final para que no fluctue de frame a frame.
    Suavizador suavCentral(24);
    Kalman1D kalmanCentral(0.03, 9.0);
    int frame_idx = 0;
    int loop_idx = 0;
    double lastRawDistanceCm = -1.0;
    double lastAcceptedDistanceCm = -1.0;
    int rejectedDistanceCount = 0;
    bool faceSwapWarned = false;
    std::vector<Detection> cachedFaces;
    cv::Mat dispTemporalF;

    while (true) {
        lastRawDistanceCm = -1.0;
        // Copia el ultimo frame publicado por cada hilo de captura.
        bool okL = camLeft.connected  && !camLeft.frame.empty();
        bool okR = camRight.connected && !camRight.frame.empty();
        if (okL) { std::lock_guard<std::mutex> l(camLeft.mtx);  camLeft.frame.copyTo(imgLeft);   }
        if (okR) { std::lock_guard<std::mutex> l(camRight.mtx); camRight.frame.copyTo(imgRight); }

        // Si alguna camara aun no entrega imagen, mostramos una pantalla de espera.
        if (!okL || !okR) {
            cv::Mat w = cv::Mat::zeros(130, 560, CV_8UC3);
            cv::putText(w, "Esperando camaras...", {10,45},
                cv::FONT_HERSHEY_SIMPLEX, 0.85, cv::Scalar(0,200,255), 2);
            cv::putText(w, okL ? "Izquierda: OK" : "Izquierda: sin frame", {10,82},
                cv::FONT_HERSHEY_SIMPLEX, 0.65, okL ? cv::Scalar(0,255,0) : cv::Scalar(0,0,255), 2);
            cv::putText(w, okR ? "Derecha: OK" : "Derecha: sin frame", {10,112},
                cv::FONT_HERSHEY_SIMPLEX, 0.65, okR ? cv::Scalar(0,255,0) : cv::Scalar(0,0,255), 2);
            cv::imshow("Camara Izquierda", w);
            if (cv::waitKey(30) == 27) break;
            continue;
        }

        // La edad del frame permite descartar pares demasiado viejos o desfasados en el tiempo.
        double ageL = edadFrameMs(camLeft);
        double ageR = edadFrameMs(camRight);
        double syncDelta = std::abs(ageL - ageR);
        if (ageL > 700.0 || ageR > 700.0 || syncDelta > 180.0) {
            std::cerr << "[WARN] Frames desincronizados o viejos: L=" << ageL
                      << " ms, R=" << ageR << " ms, delta=" << syncDelta << " ms" << std::endl;
            if (cv::waitKey(1) == 27) break;
            continue;
        }

        // Fuerza ambas imagenes al tamano usado en la calibracion estereo.
        asegurarTamanoCalibrado(imgLeft, stereoProc.imgSize);
        asegurarTamanoCalibrado(imgRight, stereoProc.imgSize);

        cv::Mat frameLeft = imgLeft;
        cv::Mat frameRight = imgRight;
        cv::Mat wbLeft, wbRight;
        // El balance de blancos por software compensa el tinte verde al desactivar AWB en firmware.
        if (wbSoftTrack == 1) {
            balanceBlancosGrayWorld(frameLeft, wbLeft);
            balanceBlancosGrayWorld(frameRight, wbRight);
            frameLeft = wbLeft;
            frameRight = wbRight;
        }

        cv::Mat rectL, rectR;
        stereoProc.rectifyImages(frameLeft, frameRight, rectL, rectR);
        cv::Mat vistaL = frameLeft.clone();
        cv::Mat procL = rectL.clone();
        cv::Mat procR = rectR.clone();

        // Brillo/contraste solo afectan la rama de profundidad; la vista a color se conserva aparte.
        double alphaC = std::max(0.1, contTrack / 10.0);
        double betaB  = brilloTrack - 100.0;
        if (std::abs(alphaC - 1.0) > 0.05 || std::abs(betaB) > 0.1) {
            procL.convertTo(procL, -1, alphaC, betaB);
            procR.convertTo(procR, -1, alphaC, betaB);
        }

        // El sistema elige automaticamente el rango de disparidad segun distancia estable o modo cercano.
        double referenciaAutoCm = kalmanCentral.iniciado ? kalmanCentral.x : lastAcceptedDistanceCm;
        numDispAuto = (modoCercaTrack == 1)
            ? 2
            : elegirModoDisparidadAuto(referenciaAutoCm, numDispAuto);
        stereoProc.setSGBMParameters(numDispAuto, blockSz);
        
        cv::Mat dispFilt, dispF;
        // SGBM + WLS producen disparidad filtrada en float, lista para visualizacion y reproyeccion 3D.
        stereoProc.computeDisparity(procL, procR, claheTrack, dispFilt, dispF);
        if (!dispF.empty()) {
            // Suavizado temporal: amortigua jitter sin recalcular nada sobre frames antiguos.
            if (dispTemporalF.empty() || dispTemporalF.size() != dispF.size()) {
                dispF.copyTo(dispTemporalF);
            } else {
                cv::addWeighted(dispF, 0.32, dispTemporalF, 0.68, 0.0, dispTemporalF);
                dispTemporalF.copyTo(dispF);
            }
        }
        cv::Mat points3D;
        if (!stereoProc.qMatrix.empty()) {
            // La matriz Q transforma cada pixel + disparidad en un punto 3D (X, Y, Z).
            cv::reprojectImageTo3D(dispF, points3D, stereoProc.qMatrix, true);
        }
        double areaMin = minAreaTrack * 100.0;
        cv::Mat confidenceMap = stereoProc.getConfidenceMap();
        
        ++loop_idx;
        // YOLO no se ejecuta en todos los frames para no matar el rendimiento.
        if (arYoloTrack == 1 && (loop_idx % 6 == 1 || cachedFaces.empty())) {
            cachedFaces = prepararRostrosParaAR(faceDetector.detect(vistaL, 0.35f), vistaL.size());
        }
        if (arYoloTrack == 0) cachedFaces.clear();
        std::vector<Detection> faces = cachedFaces;
        cv::Rect objetoVisual = detectarObjetoOscuroRectangular(vistaL, areaMin, faces);

        // Primero se forma una mascara de disparidades "basicamente validas".
        cv::Mat validDispMask = (dispF > 2.0f) & (dispF < 200.0f);
        cv::Mat confidenceMask;
        if (!confidenceMap.empty() && confidenceMap.size() == dispF.size()) {
            // Si el WLS entrega confianza util en el centro, la usamos para endurecer la mascara.
            cv::inRange(confidenceMap, cv::Scalar(70.0f), cv::Scalar(255.0f), confidenceMask);
            cv::Mat strictMask = validDispMask & confidenceMask;
            cv::Rect centroPreview = roiCentralMedicion(stereoProc.imgSize);
            if (cv::countNonZero(strictMask(centroPreview)) >= 24) {
                validDispMask = strictMask;
            }
        }
        cv::Mat confDispMask = mascaraConfianzaDisparidad(dispF, validDispMask);
        validDispMask &= confDispMask;
        // Limpiamos bordes porque suelen contener errores geometricos de rectificacion/disparidad.
        int borderX = std::max(12, stereoProc.imgSize.width / 28);
        int borderY = std::max(8, stereoProc.imgSize.height / 40);
        validDispMask(cv::Rect(0, 0, borderX, validDispMask.rows)).setTo(0);
        validDispMask(cv::Rect(validDispMask.cols - borderX, 0, borderX, validDispMask.rows)).setTo(0);
        validDispMask(cv::Rect(0, 0, validDispMask.cols, borderY)).setTo(0);
        validDispMask(cv::Rect(0, validDispMask.rows - borderY, validDispMask.cols, borderY)).setTo(0);
        cv::Mat canvas = vistaL.clone();
        if (arYoloTrack == 1) {
            std::vector<cv::Rect> rostrosSwap = seleccionarRostrosSwap(faces);
            bool faceSwapActivo = false;
            if (rostrosSwap.size() == 2) {
                try {
                    faceSwapper.swapFaces(canvas, rostrosSwap[0], rostrosSwap[1]);
                    faceSwapActivo = true;
                } catch (const cv::Exception& e) {
                    if (!faceSwapWarned) {
                        std::cerr << "[WARN] FaceSwap no pudo aplicarse con las cajas YOLO actuales: "
                                  << e.what() << std::endl;
                        faceSwapWarned = true;
                    }
                    for (const auto& f : faces) {
                        cv::rectangle(canvas, f.box, cv::Scalar(0, 255, 0), 2);
                    }
                }
            } else {
                for (const auto& f : faces) {
                    cv::rectangle(canvas, f.box, cv::Scalar(0, 255, 0), 2);
                }
            }

            const std::string faceStatus = faceSwapActivo
                ? "FaceSwap YOLO: ON"
                : "FaceSwap YOLO: " + std::to_string(faces.size()) + "/2 rostros";
            cv::putText(canvas, faceStatus, {12, 24}, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        faceSwapActivo ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 200, 255),
                        2, cv::LINE_AA);
        }

        cv::Rect objetoCentral;
        double distanciaCentralRawCm = -1.0;

        // La region central es la ROI principal de medicion exigida por la rubrica.
        cv::Rect centro = roiCentralMedicion(stereoProc.imgSize);
        cv::Mat objetoCentroMask = mascaraPrimerPlanoCentral(dispF, centro, validDispMask);
        cv::Rect objetoCentroBox = cajaDesdeMascara(objetoCentroMask, 45);
        float dispCentroPx = percentilDisparidadConMascara(dispF, centro, validDispMask, 0.78);
        if (dispCentroPx <= 0) dispCentroPx = medianaDisparidadConMascara(dispF, centro, validDispMask);
        int numDispActual = numDispAuto == 0 ? 64 : (numDispAuto == 1 ? 128 : 192);
        double distanciaCentro = -1.0;
        if (!objetoCentroBox.empty()) {
            // La distancia robusta combina Z de Q y formula estereo f*B/d dentro del mismo ROI.
            distanciaCentro = distanciaRobustaCm(
                dispF, points3D, objetoCentroBox, objetoCentroMask,
                stereoProc.focal_px, stereoProc.baseline_mm);
        }

        if (distanciaCentro > 0) {
            objetoCentral = objetoCentroBox;
            distanciaCentralRawCm = distanciaCentro;
        } else if (modoCercaTrack == 0 && !points3D.empty()) {
            // Fallback: si el centro no basto, se intenta encontrar un blob cercano por profundidad.
            cv::Rect depthBox;
            double depthMeanCm = -1.0;
            double depthThreshCm = std::clamp(static_cast<double>(depthThreshTrack), 20.0, 300.0);
            if (detectarObjetoPorProfundidad(points3D, depthThreshCm, areaMin, depthBox, depthMeanCm)) {
                objetoCentral = depthBox;
                distanciaCentralRawCm = distanciaRobustaCm(
                    dispF, points3D, depthBox, validDispMask,
                    stereoProc.focal_px, stereoProc.baseline_mm);
                if (distanciaCentralRawCm <= 0) distanciaCentralRawCm = depthMeanCm;
            }
        }

        if (modoCercaTrack == 0 && objetoCentral.empty() && !objetoVisual.empty()) {
            // Segundo fallback: usa un objeto rectangular visual si la profundidad sola no segmenta bien.
            objetoCentral = objetoVisual;
            distanciaCentralRawCm = distanciaRobustaCm(
                dispF, points3D, objetoVisual, validDispMask,
                stereoProc.focal_px, stereoProc.baseline_mm);
        }

        cv::Point centerPt(centro.x + centro.width / 2, centro.y + centro.height / 2);
        cv::drawMarker(canvas, centerPt, cv::Scalar(255, 180, 0), cv::MARKER_CROSS, 16, 1, cv::LINE_AA);
        numDispActual = numDispAuto == 0 ? 64 : (numDispAuto == 1 ? 128 : 192);
        double zMinCm = (stereoProc.focal_px * stereoProc.baseline_mm) /
                        std::max(1, numDispActual) / 10.0;
        char diagBuf[96];
        snprintf(diagBuf, sizeof(diagBuf), "Disp P78: %.1f px | %s %d | Zmin %.0f cm",
                 dispCentroPx, modoCercaTrack == 1 ? "CercaDisp" : "AutoDisp", numDispActual, zMinCm);
        cv::putText(canvas, diagBuf, {12, canvas.rows - 14},
                    cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(255, 220, 120), 2, cv::LINE_AA);

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
                // Rechaza saltos imposibles antes de alimentar media movil y Kalman.
                double referenciaCm = kalmanCentral.iniciado ? kalmanCentral.x : lastAcceptedDistanceCm;
                bool plausible = medicionDistanciaPlausible(distanciaCentralRawCm, referenciaCm);
                if (!plausible) {
                    ++rejectedDistanceCount;
                    if (rejectedDistanceCount < 4 && lastAcceptedDistanceCm > 0) {
                        distanciaCentralRawCm = lastAcceptedDistanceCm;
                    } else {
                        suavCentral.hist.clear();
                        kalmanCentral.iniciado = false;
                        kalmanCentral.x = 0.0;
                        kalmanCentral.p = 1.0;
                        rejectedDistanceCount = 0;
                    }
                } else {
                    rejectedDistanceCount = 0;
                }

                lastRawDistanceCm = distanciaCentralRawCm;
                lastAcceptedDistanceCm = distanciaCentralRawCm;

                // La salida final mostrada al usuario es Z suavizada y escalada a centimetros reales.
                double zSuavizadoRawCm = suavCentral.agregar(distanciaCentralRawCm);
                double zKalmanRawCm = kalmanCentral.actualizar(zSuavizadoRawCm);
                double zDisplayCm = zKalmanRawCm * stereoProc.scaleFactor;
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
        }

        cv::Mat dispVis, dispNorm;
        double dMin = 0.0, dMax = 0.0;
        // Para mostrar bien el mapa, se normaliza usando percentiles y no min/max crudos.
        cv::Mat dispBasicMask = (dispF > 1.0f) & (dispF < 200.0f);
        cv::Mat dispVisMask = dispBasicMask.clone();
        cv::Mat kernelVis = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::morphologyEx(dispVisMask, dispVisMask, cv::MORPH_OPEN, kernelVis);
        cv::morphologyEx(dispVisMask, dispVisMask, cv::MORPH_CLOSE, kernelVis);
        dispVisMask = filtrarComponentesPequenos(dispVisMask, 260);
        if (static_cast<size_t>(cv::countNonZero(dispVisMask)) < dispF.total() / 80) {
            dispVisMask = dispBasicMask;
            cv::morphologyEx(dispVisMask, dispVisMask, cv::MORPH_CLOSE, kernelVis);
        }
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
        auto claheDisp = cv::createCLAHE(1.0, cv::Size(16, 16));
        claheDisp->apply(dispNorm, dispNorm);
        cv::Mat dispDisplayGray = dispNorm;
        if (numDispActual > 0 && numDispActual < dispNorm.cols - 16) {
            // La banda izquierda invalida por SGBM se recorta solo para visualizacion y luego se reescala.
            cv::Rect validDispRoi(numDispActual, 0, dispNorm.cols - numDispActual, dispNorm.rows);
            cv::Mat croppedDisp = dispNorm(validDispRoi);
            cv::resize(croppedDisp, dispDisplayGray, dispNorm.size(), 0.0, 0.0, cv::INTER_LINEAR);
        }
        cv::cvtColor(dispDisplayGray, dispVis, cv::COLOR_GRAY2BGR);

        cv::Mat combined;
        std::vector<cv::Mat> panels = {canvas, dispVis};
        cv::hconcat(panels, combined);

        // Mostrar FPS de las cámaras en la esquina superior derecha
        double fpsL = camLeft.fps.load(std::memory_order_relaxed);
        double fpsR = camRight.fps.load(std::memory_order_relaxed);
        char fpsBuf[64];
        snprintf(fpsBuf, sizeof(fpsBuf), "FPS L: %.1f  R: %.1f", fpsL, fpsR);
        int fbBase = 0;
        cv::Size fsz = cv::getTextSize(fpsBuf, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &fbBase);
        int pad = 6;
        cv::rectangle(canvas,
            cv::Point(canvas.cols - fsz.width - pad - 6, 6),
            cv::Point(canvas.cols - 6, 6 + fsz.height + fbBase + 4),
            cv::Scalar(0, 0, 0), -1);
        cv::putText(canvas, fpsBuf, cv::Point(canvas.cols - fsz.width - 10, 6 + fsz.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 220, 255), 2, cv::LINE_AA);

        cv::imshow("Camara Izquierda", canvas);
        cv::imshow("Mapa de Disparidad", dispVis);
        if (recordTrack == 1 && videoOut.isOpened()) videoOut.write(combined);

        int key = cv::waitKey(1);
        if (key == 'c' || key == 'C') {
            if (lastRawDistanceCm > 0) {
                // Calibracion manual: corrige la escala global comparando medida real vs. medida calculada.
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
            // Guarda exactamente lo que ve el usuario: vista a color anotada y mapa de disparidad mostrado.
            std::string n = "frame_" + std::to_string(frame_idx++);
            cv::imwrite(n + "_vision.png", canvas);
            cv::imwrite(n + "_depth.png",  dispVis);
            std::cout << "[INFO] Guardado: " << n << std::endl;
        }
    }

    // Cierre ordenado: detiene captura, espera ambos hilos y cierra ventanas.
    camLeft.running  = false;
    camRight.running = false;
    if (threadLeft.joinable())  threadLeft.join();
    if (threadRight.joinable()) threadRight.join();
    cv::destroyAllWindows();
    return 0;
}
