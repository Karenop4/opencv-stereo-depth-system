#include "DepthUtils.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace depth {

namespace {

/**
 * @brief Expande un ROI sin salir de los límites de la imagen.
 * @param roi Región original.
 * @param bounds Dimensiones máximas de la imagen.
 * @param factor Factor de expansión aplicado al ROI.
 * @return ROI expandido y recortado a la imagen.
 * @note Se usa como recuperación cuando el centro del objeto contiene muchos
 * píxeles de profundidad inválidos.
 */
cv::Rect expandirROIClamped(const cv::Rect& roi, const cv::Size& bounds, double factor) {
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
double medianaZValidaImpl(const cv::Mat& points3D, const cv::Rect& roi, bool permitirExpansion, int profundidad) {
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

} // namespace

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
bool rangoDisparidadPercentil(const cv::Mat& dispF, const cv::Mat& validMask, double pLow, double pHigh, double& low, double& high) {
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

float medianaDisparidadConMascara(const cv::Mat& dispF, const cv::Rect& roi, const cv::Mat& mask);
float percentilDisparidadConMascara(const cv::Mat& dispF, const cv::Rect& roi, const cv::Mat& mask, double percentile);

/**
 * @brief Genera una máscara de confianza de disparidad a partir de la mediana local.
 * @param dispF Mapa de disparidad en float.
 * @param validMask Máscara base de disparidades válidas.
 * @return Máscara binaria de píxeles confiables.
 * @note Reduce speckle noise y evita que outliers contaminen la distancia y la visualización.
 */
cv::Mat mascaraConfianzaDisparidad(const cv::Mat& dispF, const cv::Mat& validMask) {
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
 * @brief Construye una máscara de primer plano dentro de la ROI central.
 * @param dispF Mapa de disparidad.
 * @param roi ROI de búsqueda.
 * @param validMask Máscara de disparidades válidas.
 * @return Máscara global del componente cercano más centrado.
 * @note Evita que la distancia central se vaya al fondo cuando el objeto no
 * llena toda la ROI, una causa común de lecturas aparentemente invertidas.
 */
cv::Mat mascaraPrimerPlanoCentral(const cv::Mat& dispF, const cv::Rect& roi, const cv::Mat& validMask) {
    cv::Mat out = cv::Mat::zeros(dispF.size(), CV_8U);
    cv::Rect safe = roi & cv::Rect(0, 0, dispF.cols, dispF.rows);
    if (safe.empty()) return out;

    float p60 = percentilDisparidadConMascara(dispF, safe, validMask, 0.60);
    float p90 = percentilDisparidadConMascara(dispF, safe, validMask, 0.90);
    if (p60 <= 0 || p90 <= 0 || p90 <= p60) return out;

    float threshold = p60 + 0.20f * (p90 - p60);
    cv::Mat localDisp = dispF(safe);
    cv::Mat localValid = validMask(safe);
    cv::Mat localMask = (localDisp >= threshold) & localValid;

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(localMask, localMask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(localMask, localMask, cv::MORPH_CLOSE, kernel);

    cv::Mat labels, stats, centroids;
    int n = cv::connectedComponentsWithStats(localMask, labels, stats, centroids, 8, CV_32S);
    if (n <= 1) return out;

    cv::Point2d center(safe.width * 0.5, safe.height * 0.5);
    int bestLabel = -1;
    double bestScore = -1.0;
    for (int label = 1; label < n; ++label) {
        int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area < std::max(40, safe.area() / 80)) continue;
        cv::Point2d c(centroids.at<double>(label, 0), centroids.at<double>(label, 1));
        double dist = cv::norm(c - center) / std::max(safe.width, safe.height);
        double score = area / (1.0 + 5.0 * dist);
        if (score > bestScore) {
            bestScore = score;
            bestLabel = label;
        }
    }

    if (bestLabel < 0) return out;
    cv::Mat selected = labels == bestLabel;
    selected.copyTo(out(safe));
    return out;
}

/**
 * @brief Elimina componentes pequeños de una máscara binaria.
 * @param mask Máscara de entrada.
 * @param minArea Área mínima de componente.
 * @return Máscara filtrada.
 * @note Reduce speckles visuales que no representan superficies coherentes.
 */
cv::Mat filtrarComponentesPequenos(const cv::Mat& mask, int minArea) {
    cv::Mat clean = cv::Mat::zeros(mask.size(), CV_8U);
    if (mask.empty()) return clean;

    cv::Mat labels, stats, centroids;
    int n = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);
    for (int label = 1; label < n; ++label) {
        int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area >= minArea) clean.setTo(255, labels == label);
    }
    return clean;
}

/**
 * @brief Devuelve la caja envolvente de una máscara si tiene área suficiente.
 * @param mask Máscara global del objeto.
 * @param minArea Área mínima requerida.
 * @return Bounding box o rectángulo vacío.
 * @note Se usa para dibujar solo objetos detectados, no la ROI fija central.
 */
cv::Rect cajaDesdeMascara(const cv::Mat& mask, int minArea) {
    std::vector<cv::Point> pts;
    cv::findNonZero(mask, pts);
    if (static_cast<int>(pts.size()) < minArea) return {};
    return cv::boundingRect(pts);
}

/**
 * @brief Calcula la mediana de disparidad sobre un ROI usando una máscara de confianza.
 * @param dispF Mapa de disparidad en float.
 * @param roi Región de interés.
 * @param mask Máscara de píxeles confiables.
 * @return Mediana de disparidad o -1 si no hay muestras válidas.
 * @note Esta medición es la base de la conversión a centímetros sin contaminarse por speckles.
 */
float medianaDisparidadConMascara(const cv::Mat& dispF, const cv::Rect& roi, const cv::Mat& mask) {
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
 * @brief Percentil alto de disparidad para priorizar el objeto cercano en la ROI.
 * @param dispF Mapa de disparidad en píxeles.
 * @param roi Región de medición.
 * @param mask Máscara de disparidades válidas.
 * @param percentile Percentil en rango 0..1.
 * @return Disparidad robusta en píxeles o -1 si no hay muestras.
 * @note En una ROI central puede mezclarse objeto y fondo; un percentil alto
 * toma el primer plano, que es lo esperado para medir el objeto colocado al centro.
 */
float percentilDisparidadConMascara(const cv::Mat& dispF, const cv::Rect& roi, const cv::Mat& mask, double percentile) {
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
    size_t idx = std::min(vals.size() - 1,
        static_cast<size_t>(std::clamp(percentile, 0.0, 1.0) * (vals.size() - 1)));
    std::nth_element(vals.begin(), vals.begin() + idx, vals.end());
    return vals[idx];
}

/**
 * @brief Calcula la mediana Z válida usando la función auxiliar con expansión.
 * @param points3D Nube de puntos reproyectada.
 * @param roi Región de interés.
 * @return Mediana Z en milímetros o -1 si no hay datos válidos.
 * @note Es el valor métrico que alimenta la estabilización temporal y la calibración.
 */
double medianaZValida(const cv::Mat& points3D, const cv::Rect& roi) {
    return medianaZValidaImpl(points3D, roi, true, 0);
}

/**
 * @brief ROI central usada para medir el objeto situado frente al sistema.
 * @param size Tamaño del frame rectificado.
 * @return Caja central proporcional al campo de visión.
 * @note La rúbrica pide enfocar la métrica en el objeto colocado en el centro,
 * por eso esta región tiene prioridad sobre detectores visuales auxiliares.
 */
cv::Rect roiCentralMedicion(const cv::Size& size) {
    int w = std::max(70, size.width / 8);
    int h = std::max(60, size.height / 8);
    return cv::Rect((size.width - w) / 2, (size.height - h) / 2, w, h);
}

/**
 * @brief Elige automáticamente el rango de disparidad necesario.
 * @param distanciaEstableCm Última distancia filtrada disponible.
 * @param modoActual Modo actual: 0=64, 1=128, 2=192.
 * @return Nuevo modo con histéresis para evitar cambios constantes.
 * @note Permite medir cerca sin que el usuario tenga que cambiar NumDisp,
 * pero baja el coste cuando el objeto está lejos y estable.
 */
int elegirModoDisparidadAuto(double distanciaEstableCm, int modoActual) {
    if (!std::isfinite(distanciaEstableCm) || distanciaEstableCm <= 0.0) {
        return 1;
    }

    if (modoActual == 2) {
        if (distanciaEstableCm > 44.0) return 1;
        return 2;
    }
    if (modoActual == 1) {
        if (distanciaEstableCm < 32.0) return 2;
        if (distanciaEstableCm > 145.0) return 0;
        return 1;
    }
    if (distanciaEstableCm < 115.0) return 1;
    return 0;
}

/**
 * @brief Rechaza saltos de distancia incompatibles con un objeto estático.
 * @param medicionCm Medición nueva.
 * @param referenciaCm Última distancia aceptada o filtrada.
 * @return true si la medición se considera plausible.
 * @note Evita fluctuaciones grandes causadas por speckles o cambios de ROI.
 */
bool medicionDistanciaPlausible(double medicionCm, double referenciaCm) {
    if (!std::isfinite(medicionCm) || medicionCm <= 0.0) return false;
    if (!std::isfinite(referenciaCm) || referenciaCm <= 0.0) return true;

    double saltoMax = std::max(7.0, 0.10 * referenciaCm);
    return std::abs(medicionCm - referenciaCm) <= saltoMax;
}

/**
 * @brief Calcula distancia robusta dentro de una ROI combinando Q y disparidad.
 * @param dispF Mapa de disparidad en píxeles.
 * @param points3D Nube reproyectada con la matriz Q.
 * @param roi Región a medir.
 * @param validDispMask Máscara de disparidades confiables.
 * @param focalPx Focal de la cámara rectificada.
 * @param baselineMm Línea base en milímetros.
 * @return Distancia en centímetros o -1 si no existe lectura válida.
 * @note Mantiene explícita la reproyección 3D con Q y usa la fórmula de
 * disparidad como comprobación robusta ante huecos locales.
 */
double distanciaRobustaCm(
    const cv::Mat& dispF,
    const cv::Mat& points3D,
    const cv::Rect& roi,
    const cv::Mat& validDispMask,
    double focalPx,
    double baselineMm)
{
    cv::Rect safe = roi & cv::Rect(0, 0, dispF.cols, dispF.rows);
    if (safe.empty()) return -1.0;

    cv::Rect inner(
        safe.x + safe.width / 6,
        safe.y + safe.height / 6,
        std::max(1, safe.width * 2 / 3),
        std::max(1, safe.height * 2 / 3));

    cv::Mat fgMask = mascaraPrimerPlanoCentral(dispF, safe, validDispMask);
    const cv::Mat& measureMask = (cv::countNonZero(fgMask(safe)) >= 8) ? fgMask : validDispMask;

    float dispMed = percentilDisparidadConMascara(dispF, inner, measureMask, 0.65);
    if (dispMed <= 0) dispMed = percentilDisparidadConMascara(dispF, safe, measureMask, 0.65);
    if (dispMed <= 0) dispMed = medianaDisparidadConMascara(dispF, inner, validDispMask);
    if (dispMed <= 0) dispMed = medianaDisparidadConMascara(dispF, safe, validDispMask);

    double zFromDispCm = -1.0;
    if (dispMed > 0) {
        zFromDispCm = (focalPx * baselineMm) / dispMed / 10.0;
    }

    double zFromPointsCm = -1.0;
    if (!points3D.empty()) {
        zFromPointsCm = medianaZValida(points3D, inner);
        if (zFromPointsCm <= 0) zFromPointsCm = medianaZValida(points3D, safe);
        if (zFromPointsCm > 0) zFromPointsCm /= 10.0;
    }

    if (zFromDispCm > 0 && zFromPointsCm > 0) {
        double relDiff = std::abs(zFromDispCm - zFromPointsCm) / std::max(1.0, zFromDispCm);
        return (relDiff > 0.35) ? zFromDispCm : 0.5 * (zFromDispCm + zFromPointsCm);
    }
    if (zFromDispCm > 0) return zFromDispCm;
    return zFromPointsCm;
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
bool detectarObjetoPorProfundidad(
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

} // namespace depth
