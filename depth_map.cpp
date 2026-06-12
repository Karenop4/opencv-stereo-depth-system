#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <numeric>

// Bloque ajustado con asistencia de IA: usa GStreamer como primera opcion de
// captura para mantener el calibrador alineado con la aplicacion principal.
std::string shell_quote_gst_value(const std::string& value) {
    std::string quoted = "'";
    for (char ch : value) {
        if (ch == '\'') {
            quoted += "\\'";
        } else {
            quoted += ch;
        }
    }
    quoted += "'";
    return quoted;
}

std::string build_gstreamer_pipeline(const std::string& url) {
    if (url.rfind("http://", 0) == 0 || url.rfind("https://", 0) == 0) {
        return "souphttpsrc location=" + shell_quote_gst_value(url) +
               " is-live=true do-timestamp=true timeout=3 ! "
               "multipartdemux ! jpegdec ! videoconvert ! "
               "appsink drop=true max-buffers=1 sync=false";
    }

    if (url.rfind("/dev/video", 0) == 0) {
        return "v4l2src device=" + shell_quote_gst_value(url) +
               " ! videoconvert ! appsink drop=true max-buffers=1 sync=false";
    }

    return {};
}

bool open_capture(cv::VideoCapture& cap, const std::string& url) {
    const std::string gstPipeline = build_gstreamer_pipeline(url);
    if (!gstPipeline.empty() && cap.open(gstPipeline, cv::CAP_GSTREAMER)) {
        std::cout << "[CAM] Backend GStreamer activo." << std::endl;
        return true;
    }
    cap.release();

    if (cap.open(url, cv::CAP_FFMPEG)) return true;
    cap.release();
    return cap.open(url, cv::CAP_ANY);
}

// ─────────────────────────────────────────────
//  Estructura de stream por cámara
// ─────────────────────────────────────────────
struct CameraStream {
    std::string url;
    cv::Mat frame;
    std::mutex mtx;
    std::atomic<bool> running;
    std::atomic<bool> connected;

    CameraStream(std::string camera_url)
        : url(camera_url), running(true), connected(false) {}
};

// ─────────────────────────────────────────────
//  Hilo de captura con reconexión automática
// ─────────────────────────────────────────────
void capture_loop(CameraStream* cam) {
    while (cam->running) {
        cv::VideoCapture cap;
        open_capture(cap, cam->url);
        if (!cap.isOpened()) {
            cam->connected = false;
            std::this_thread::sleep_for(std::chrono::seconds(2));
            continue;
        }
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
        cam->connected = true;

        cv::Mat tmp_frame;
        while (cam->running) {
            if (!cap.grab()) break;
            cap.retrieve(tmp_frame);
            if (tmp_frame.empty()) continue;
            cv::resize(tmp_frame, tmp_frame, cv::Size(640, 480), 0, 0, cv::INTER_LINEAR);
            {
                std::lock_guard<std::mutex> lock(cam->mtx);
                tmp_frame.copyTo(cam->frame);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        cap.release();
        cam->connected = false;
    }
}

// ─────────────────────────────────────────────
//  Filtro: rechaza muestras demasiado parecidas
// ─────────────────────────────────────────────
bool esMuestraDiversa(
    const std::vector<std::vector<cv::Point2f>>& capturas,
    const std::vector<cv::Point2f>& nueva,
    float umbralPixeles = 25.0f)
{
    for (const auto& anterior : capturas) {
        float dist = 0.0f;
        for (size_t i = 0; i < anterior.size(); i++)
            dist += (float)cv::norm(anterior[i] - nueva[i]);
        dist /= (float)anterior.size();
        if (dist < umbralPixeles) return false;
    }
    return true;
}

// ─────────────────────────────────────────────
//  Centroide de un conjunto de puntos 2D
// ─────────────────────────────────────────────
cv::Point2f centroide(const std::vector<cv::Point2f>& pts) {
    cv::Point2f c(0.0f, 0.0f);
    for (const auto& p : pts) c += p;
    return c * (1.0f / (float)pts.size());
}

double movimientoMedio(const std::vector<cv::Point2f>& a, const std::vector<cv::Point2f>& b) {
    if (a.size() != b.size() || a.empty()) return 1e9;
    double acc = 0.0;
    for (size_t i = 0; i < a.size(); ++i) acc += cv::norm(a[i] - b[i]);
    return acc / static_cast<double>(a.size());
}

bool detectarTablero(const cv::Mat& img, const cv::Size& patternSize, std::vector<cv::Point2f>& corners) {
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    int sbFlags = cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_EXHAUSTIVE | cv::CALIB_CB_ACCURACY;
    bool found = cv::findChessboardCornersSB(gray, patternSize, corners, sbFlags);
    if (found) return true;

    int flags = cv::CALIB_CB_ADAPTIVE_THRESH
              | cv::CALIB_CB_NORMALIZE_IMAGE
              | cv::CALIB_CB_FILTER_QUADS;
    found = cv::findChessboardCorners(gray, patternSize, corners, flags);
    if (found) {
        cv::TermCriteria sp(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 100, 0.001);
        cv::cornerSubPix(gray, corners, {11, 11}, {-1, -1}, sp);
    }
    return found;
}

// ─────────────────────────────────────────────
//  Calibración estéreo completa para ESP32-CAM
// ─────────────────────────────────────────────
void ejecutarCalibracion(
    const std::vector<std::vector<cv::Point3f>>& objectPoints,
    const std::vector<std::vector<cv::Point2f>>& imagePointsLeft,
    const std::vector<std::vector<cv::Point2f>>& imagePointsRight,
    cv::Size imageSize)
{
    if (objectPoints.size() != imagePointsLeft.size() ||
        objectPoints.size() != imagePointsRight.size() ||
        objectPoints.empty()) {
        std::cerr << "[ERROR] Vectores inconsistentes o vacíos." << std::endl;
        return;
    }

    std::cout << "\n==========================================================" << std::endl;
    std::cout << "[INFO] Paso 1: Calibración individual por cámara..." << std::endl;
    std::cout << "==========================================================" << std::endl;

    cv::Mat K1 = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat K2 = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat D1 = cv::Mat::zeros(1, 5, CV_64F);
    cv::Mat D2 = cv::Mat::zeros(1, 5, CV_64F);

    int flags_ind = cv::CALIB_FIX_K3;

    std::vector<cv::Mat> rvecsL, tvecsL, rvecsR, tvecsR;

    double rmsL = cv::calibrateCamera(objectPoints, imagePointsLeft,
        imageSize, K1, D1, rvecsL, tvecsL, flags_ind);
    double rmsR = cv::calibrateCamera(objectPoints, imagePointsRight,
        imageSize, K2, D2, rvecsR, tvecsR, flags_ind);

    std::cout << "[RMS individual] Izquierda=" << rmsL
              << "  Derecha=" << rmsR << std::endl;
    std::cout << "[INFO] Coeficientes D1: " << D1.cols
              << "  D2: " << D2.cols << std::endl;

    // ── Paso 2: Detectar y eliminar outliers ──────────────────────────────
    std::cout << "\n[INFO] Paso 2: Filtrando outliers..." << std::endl;

    std::vector<int> indicesValidos;
    for (int i = 0; i < (int)objectPoints.size(); i++) {
        std::vector<cv::Point2f> projL, projR;
        cv::projectPoints(objectPoints[i], rvecsL[i], tvecsL[i], K1, D1, projL);
        cv::projectPoints(objectPoints[i], rvecsR[i], tvecsR[i], K2, D2, projR);
        double errL = cv::norm(imagePointsLeft[i],  projL, cv::NORM_L2) / sqrt((double)projL.size());
        double errR = cv::norm(imagePointsRight[i], projR, cv::NORM_L2) / sqrt((double)projR.size());
        double errMax = std::max(errL, errR);

        if (errMax < 1.2) {
            indicesValidos.push_back(i);
        } else {
            std::cout << "  [DESCARTADA] Muestra #" << i
                      << "  errL=" << errL << "  errR=" << errR << std::endl;
        }
    }

    std::cout << "[INFO] Muestras válidas: "
              << indicesValidos.size() << " / " << objectPoints.size() << std::endl;

    if ((int)indicesValidos.size() < 8) {
        std::cerr << "[ERROR] Solo " << indicesValidos.size()
                  << " muestras válidas — necesitas al menos 8." << std::endl;
        std::cerr << "        Recalibra con mejor iluminación y posiciones más diversas." << std::endl;
        return;
    }

    std::vector<std::vector<cv::Point3f>> objClean;
    std::vector<std::vector<cv::Point2f>> imgLClean, imgRClean;
    for (int idx : indicesValidos) {
        objClean.push_back(objectPoints[idx]);
        imgLClean.push_back(imagePointsLeft[idx]);
        imgRClean.push_back(imagePointsRight[idx]);
    }

    // ── Paso 3: Calibración estéreo ───────────────────────────────────────
    std::cout << "\n[INFO] Paso 3: Calibración estéreo completa..." << std::endl;

    cv::Mat R, T, E, F;
    cv::TermCriteria criteria(
        cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 300, 1e-7);

    double rmsPreliminar = cv::stereoCalibrate(
        objClean, imgLClean, imgRClean,
        K1, D1, K2, D2, imageSize, R, T, E, F,
        cv::CALIB_FIX_INTRINSIC,
        criteria);

    std::cout << "[INFO] RMS estereo preliminar: " << rmsPreliminar << std::endl;

    std::vector<std::vector<cv::Point3f>> objStereo;
    std::vector<std::vector<cv::Point2f>> imgLStereo, imgRStereo;
    for (int i = 0; i < (int)objClean.size(); ++i) {
        std::vector<cv::Vec3f> linesR;
        cv::computeCorrespondEpilines(imgLClean[i], 1, F, linesR);

        double err = 0.0;
        for (size_t j = 0; j < imgRClean[i].size(); ++j) {
            const cv::Vec3f& l = linesR[j];
            const cv::Point2f& p = imgRClean[i][j];
            err += std::abs(l[0] * p.x + l[1] * p.y + l[2]) /
                   std::sqrt(l[0] * l[0] + l[1] * l[1] + 1e-12);
        }
        err /= imgRClean[i].size();

        if (err < 2.5) {
            objStereo.push_back(objClean[i]);
            imgLStereo.push_back(imgLClean[i]);
            imgRStereo.push_back(imgRClean[i]);
        } else {
            std::cout << "  [DESCARTADA ESTEREO] Muestra limpia #" << i
                      << "  error epipolar=" << err << "px" << std::endl;
        }
    }

    if ((int)objStereo.size() < 8) {
        std::cerr << "[WARN] El filtro epipolar dejo pocas muestras; se usan las muestras limpias iniciales." << std::endl;
        objStereo = objClean;
        imgLStereo = imgLClean;
        imgRStereo = imgRClean;
    }

    double rms = cv::stereoCalibrate(
        objStereo, imgLStereo, imgRStereo,
        K1, D1, K2, D2, imageSize, R, T, E, F,
        cv::CALIB_FIX_INTRINSIC,
        criteria);

    std::cout << "\n==========================================================" << std::endl;
    std::cout << "[RESULTADO] RMS estéreo: " << rms << std::endl;
    std::cout << "[T] baseline = " << abs(T.at<double>(0)) << " mm"
              << "   desv. vertical = " << T.at<double>(1) << " mm"
              << "   desv. profundidad = " << T.at<double>(2) << " mm" << std::endl;
    double baselineNorm = cv::norm(T);
    std::cout << "[T] baseline norma = " << baselineNorm << " mm" << std::endl;

    if (abs(T.at<double>(1)) > 5.0)
        std::cout << "[ADVERTENCIA] Desalineación vertical > 5mm — ajusta la altura del hardware." << std::endl;
    if (abs(T.at<double>(2)) > 5.0)
        std::cout << "[ADVERTENCIA] Una cámara está más adelantada que la otra > 5mm." << std::endl;

    if      (rms < 0.5) std::cout << "[CALIDAD] Excelente (< 0.5)" << std::endl;
    else if (rms < 1.0) std::cout << "[CALIDAD] Buena (< 1.0)" << std::endl;
    else if (rms < 2.0) std::cout << "[CALIDAD] Aceptable (< 2.0) — intenta más muestras en bordes" << std::endl;
    else                std::cout << "[CALIDAD] Pobre (> 2.0) — revisa hardware e iluminación" << std::endl;
    std::cout << "==========================================================" << std::endl;

    // ── Paso 4: Rectificación ──────────────────────────────────────────────
    cv::Mat R1, R2, P1, P2, Q;
    cv::Rect validRoi[2];
    cv::stereoRectify(
        K1, D1, K2, D2, imageSize, R, T,
        R1, R2, P1, P2, Q,
        cv::CALIB_ZERO_DISPARITY,
        0,
        imageSize,
        &validRoi[0], &validRoi[1]);

    bool calibracionFisicaOk = rms < 3.0 && baselineNorm > 35.0 && baselineNorm < 140.0;
    if (!calibracionFisicaOk) {
        std::cerr << "[ERROR] Calibracion rechazada: RMS/baseline no son fisicamente confiables." << std::endl;
        std::cerr << "        No se sobrescribe parametros_stereo.yml." << std::endl;
        std::cerr << "        Revisa muestras, rigidez del soporte y que el tablero este quieto." << std::endl;
    }

    // ── Paso 5: Guardar ────────────────────────────────────────────────────
    std::string filename = calibracionFisicaOk ? "parametros_stereo.yml" : "parametros_stereo_rechazado.yml";
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    if (fs.isOpened()) {
        fs << "imageSize"      << imageSize
           << "K1" << K1 << "D1" << D1
           << "K2" << K2 << "D2" << D2
           << "R"  << R  << "T"  << T
           << "E"  << E  << "F"  << F
           << "R1" << R1 << "R2" << R2
           << "P1" << P1 << "P2" << P2
           << "Q"  << Q
           << "validRoiLeft"   << validRoi[0]
           << "validRoiRight"  << validRoi[1]
           << "rmsEstereo"     << rms
           << "rmsLeft"        << rmsL
           << "rmsRight"       << rmsR
           << "muestrasUsadas" << (int)objStereo.size()
           << "muestrasTotales"<< (int)objectPoints.size();
        fs.release();
        std::cout << "[INFO] '" << filename << "' guardado correctamente." << std::endl;
    } else {
        std::cerr << "[ERROR] No se pudo guardar el archivo." << std::endl;
    }
}

// ─────────────────────────────────────────────
//  MAIN
// ─────────────────────────────────────────────
int main() {
    CameraStream camLeft ("http://192.168.3.45:81/stream");
    CameraStream camRight("http://192.168.3.44:81/stream");

    std::cout << "[INFO] Conectando hilos de captura..." << std::endl;
    std::thread threadLeft (capture_loop, &camLeft);
    std::thread threadRight(capture_loop, &camRight);

    std::string window_name = "Calibracion Estereo ESP32-CAM";
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

    // ── Parámetros del tablero ─────────────────────────────────────────────
    cv::Size patternSize(7, 7);   // tablero fisico 8x8 => 7x7 esquinas internas
    float squareSizeMM = 24.0f;   // tamaño real de cada cuadrado en mm

    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imagePointsLeft;
    std::vector<std::vector<cv::Point2f>> imagePointsRight;

    std::vector<cv::Point3f> obj;
    for (int i = 0; i < patternSize.height; ++i)
        for (int j = 0; j < patternSize.width; ++j)
            obj.push_back(cv::Point3f(j * squareSizeMM, i * squareSizeMM, 0.0f));

    std::cout << "\n[GUIA DE CAPTURA]:" << std::endl;
    std::cout << "  1. Centro del frame, tablero plano y frontal" << std::endl;
    std::cout << "  2. Esquinas del frame (4 posiciones)" << std::endl;
    std::cout << "  3. Tablero inclinado ~30 horizontal (izq y der)" << std::endl;
    std::cout << "  4. Tablero inclinado ~30 vertical (arriba y abajo)" << std::endl;
    std::cout << "  5. Cerca (~40cm) y lejos (~100cm)" << std::endl;
    std::cout << "\n[CONTROLES]:" << std::endl;
    std::cout << "  [ESPACIO] -> Capturar muestra" << std::endl;
    std::cout << "  [C]       -> Calibrar y guardar (minimo 15 muestras)" << std::endl;
    std::cout << "  [ESC]     -> Abortar" << std::endl;

    cv::Mat imgLeft, imgRight;
    cv::Mat stereo_canvas = cv::Mat::zeros(480, 1280, CV_8UC3);
    int capture_count = 0;
    std::vector<cv::Point2f> prevCornersLeft, prevCornersRight;
    double movimientoLeft = 1e9;
    double movimientoRight = 1e9;

    while (camLeft.running && camRight.running) {

        // ── Leer frames ──────────────────────────────────────────────────
        if (camLeft.connected) {
            std::lock_guard<std::mutex> lock(camLeft.mtx);
            if (!camLeft.frame.empty()) camLeft.frame.copyTo(imgLeft);
        } else {
            imgLeft = cv::Mat::zeros(480, 640, CV_8UC3);
            cv::putText(imgLeft, "Buscando .45...", cv::Point(150, 240),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
        }

        if (camRight.connected) {
            std::lock_guard<std::mutex> lock(camRight.mtx);
            if (!camRight.frame.empty()) camRight.frame.copyTo(imgRight);
        } else {
            imgRight = cv::Mat::zeros(480, 640, CV_8UC3);
            cv::putText(imgRight, "Buscando .44...", cv::Point(150, 240),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
        }

        cv::Mat viewLeft  = imgLeft.clone();
        cv::Mat viewRight = imgRight.clone();
        std::vector<cv::Point2f> cornersLeft, cornersRight;
        bool foundLeft = false, foundRight = false;

        // ── Detección de esquinas en vivo ─────────────────────────────────
        if (camLeft.connected && camRight.connected &&
            !imgLeft.empty() && !imgRight.empty())
        {
            foundLeft  = detectarTablero(imgLeft, patternSize, cornersLeft);
            foundRight = detectarTablero(imgRight, patternSize, cornersRight);

            movimientoLeft = foundLeft ? movimientoMedio(prevCornersLeft, cornersLeft) : 1e9;
            movimientoRight = foundRight ? movimientoMedio(prevCornersRight, cornersRight) : 1e9;
            if (foundLeft) prevCornersLeft = cornersLeft;
            if (foundRight) prevCornersRight = cornersRight;

            if (foundLeft)  cv::drawChessboardCorners(viewLeft,  patternSize, cornersLeft,  true);
            if (foundRight) cv::drawChessboardCorners(viewRight, patternSize, cornersRight, true);

            cv::circle(viewLeft,  {20, 20}, 10,
                foundLeft  ? cv::Scalar(0,255,0) : cv::Scalar(0,0,255), -1);
            cv::circle(viewRight, {20, 20}, 10,
                foundRight ? cv::Scalar(0,255,0) : cv::Scalar(0,0,255), -1);
        }

        // ── Ensamblar canvas side-by-side ─────────────────────────────────
        if (!viewLeft.empty() && !viewRight.empty()) {
            viewLeft.copyTo (stereo_canvas(cv::Rect(0,   0, 640, 480)));
            viewRight.copyTo(stereo_canvas(cv::Rect(640, 0, 640, 480)));
        }

        std::string info = "Muestras: " + std::to_string(capture_count) + " / 15 min";
        cv::putText(stereo_canvas, info, {30, 40},
            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

        bool ambasOk = foundLeft && foundRight;
        bool estable = ambasOk && movimientoLeft < 1.5 && movimientoRight < 1.5;
        cv::putText(stereo_canvas,
            estable ? "Tablero estable: puedes capturar" :
            (ambasOk ? "Tablero detectado: mantenlo quieto" : "Esperando tablero..."),
            {30, 470}, cv::FONT_HERSHEY_SIMPLEX, 0.6,
            estable ? cv::Scalar(0,255,100) : cv::Scalar(100,100,255), 1);

        cv::line(stereo_canvas, {640,0}, {640,479}, cv::Scalar(255,255,0), 1);

        cv::imshow(window_name, stereo_canvas);
        int key = cv::waitKey(30);

        // ── ESC: abortar ──────────────────────────────────────────────────
        if (key == 27) {
            std::cout << "[INFO] Abortado." << std::endl;
            break;
        }

        // ── ESPACIO: capturar muestra ─────────────────────────────────────
        else if (key == 32) {
            if (!foundLeft || !foundRight) {
                std::cout << "[WARN] Tablero no visible en ambas camaras." << std::endl;
                continue;
            }
            if (movimientoLeft > 1.5 || movimientoRight > 1.5) {
                std::cout << "[WARN] Tablero aun se mueve: movL=" << movimientoLeft
                          << "px movR=" << movimientoRight
                          << "px. Mantenlo quieto y vuelve a capturar." << std::endl;
                continue;
            }

            cv::Mat grayL, grayR;
            cv::cvtColor(imgLeft,  grayL, cv::COLOR_BGR2GRAY);
            cv::cvtColor(imgRight, grayR, cv::COLOR_BGR2GRAY);

            cv::TermCriteria sp(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 100, 0.001);
            
            // [CORRECCIÓN]: Se aumentó la ventana de búsqueda a {11, 11} para mayor precisión subpixel a 640x480.
            cv::cornerSubPix(grayL, cornersLeft,  {11,11}, {-1,-1}, sp);
            cv::cornerSubPix(grayR, cornersRight, {11,11}, {-1,-1}, sp);

            cv::Point2f cL = centroide(cornersLeft);
            cv::Point2f cR = centroide(cornersRight);
            float deltaY = abs(cL.y - cR.y);

            if (deltaY > 60.0f) {
                std::cout << "[RECHAZADO] Desalineacion vertical severa: "
                          << deltaY << "px. Ajusta el hardware." << std::endl;
                stereo_canvas = cv::Scalar(0, 0, 180);
                cv::imshow(window_name, stereo_canvas);
                cv::waitKey(150);
                continue;
            }

            if (!esMuestraDiversa(imagePointsLeft, cornersLeft)) {
                std::cout << "[RECHAZADO] Muestra muy similar a una anterior. Mueve mas el tablero." << std::endl;
                stereo_canvas = cv::Scalar(0, 120, 255); 
                cv::imshow(window_name, stereo_canvas);
                cv::waitKey(150);
                continue;
            }

            // Muestra aceptada
            imagePointsLeft.push_back(cornersLeft);
            imagePointsRight.push_back(cornersRight);
            objectPoints.push_back(obj);
            capture_count++;

            std::cout << "[OK] Muestra #" << capture_count
                      << "  delta centroide Y = " << deltaY << "px" << std::endl;

            stereo_canvas = cv::Scalar(0, 200, 0);
            cv::imshow(window_name, stereo_canvas);
            cv::waitKey(150);
        }

        // ── C: calibrar ───────────────────────────────────────────────────
        else if (key == 'c' || key == 'C') {
            if (capture_count < 15) {
                std::cout << "[ERROR] Necesitas al menos 15 muestras. Llevas: "
                          << capture_count << std::endl;
            } else {
                ejecutarCalibracion(
                    objectPoints, imagePointsLeft, imagePointsRight,
                    cv::Size(640, 480));
                break;
            }
        }
    }

    // ── Limpieza ──────────────────────────────────────────────────────────
    std::cout << "[INFO] Cerrando hilos..." << std::endl;
    camLeft.running  = false;
    camRight.running = false;
    if (threadLeft.joinable())  threadLeft.join();
    if (threadRight.joinable()) threadRight.join();
    cv::destroyAllWindows();
    return 0;
}
