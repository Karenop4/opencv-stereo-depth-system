#include "StereoProcessor.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>

StereoProcessor::StereoProcessor(const std::string& calibration_file, const std::string& scale_file)
    : focal_px(0.0), baseline_mm(0.0), scaleFactor(1.0), imgSize(640, 480) {
    // Abre el archivo de calibracion que contiene intrinsecos, extrinsecos y, si existe, Q ya calculada.
    cv::FileStorage fs(calibration_file, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "[ERROR] No se encontró " << calibration_file << std::endl;
        std::exit(-1);
    }

    cv::FileNode imageSizeNode = fs["imageSize"];
    if (!imageSizeNode.empty()) {
        imageSizeNode >> imgSize;
        std::cout << "[INFO] imageSize from calibration: "
                  << imgSize.width << "x" << imgSize.height << std::endl;
    }

    cv::Mat K1, K2, D1, D2, R, T, R1, R2, P1, P2, Q;
    // K = matriz intrinseca, D = distorsion, R/T = relacion entre camaras, P/Q = rectificacion/proyeccion.
    fs["K1"] >> K1; fs["D1"] >> D1;
    fs["K2"] >> K2; fs["D2"] >> D2;
    fs["R"] >> R; fs["T"] >> T;
    fs["R1"] >> R1; fs["R2"] >> R2;
    fs["P1"] >> P1; fs["P2"] >> P2;
    fs["Q"] >> Q;
    fs.release();

    if (R1.empty() || R2.empty() || P1.empty() || P2.empty() || Q.empty()) {
        // Si la rectificacion no esta guardada, la recalculamos desde los parametros estereo basicos.
        cv::Rect validRoi[2];
        cv::stereoRectify(K1, D1, K2, D2, imgSize, R, T,
            R1, R2, P1, P2, Q,
            cv::CALIB_ZERO_DISPARITY, 0, imgSize,
            &validRoi[0], &validRoi[1]);
        std::cout << "[INFO] Rectificacion calculada desde K/D/R/T" << std::endl;
    } else {
        std::cout << "[INFO] Rectificacion cargada desde calibracion" << std::endl;
    }
    qMatrix = Q.clone();

    // focal_px y baseline_mm son los dos escalares clave de la formula Z = f * B / d.
    focal_px = std::abs(P1.at<double>(0, 0));
    if (!P2.empty() && std::abs(P2.at<double>(0, 0)) > 1e-9) {
        baseline_mm = std::abs(P2.at<double>(0, 3) / P2.at<double>(0, 0));
        std::cout << "[INFO] Baseline from P2: " << baseline_mm << std::endl;
    }
    if (baseline_mm <= 0.0 && !Q.empty() && std::abs(Q.at<double>(3, 2)) > 1e-9) {
        baseline_mm = std::abs(1.0 / Q.at<double>(3, 2));
        std::cout << "[INFO] Baseline from Q fallback: " << baseline_mm << std::endl;
    }
    if (baseline_mm <= 0.0 && !T.empty() && T.total() >= 3) {
        baseline_mm = cv::norm(T);
        std::cout << "[INFO] Baseline from T fallback: " << baseline_mm << std::endl;
    }
    std::cout << "[INFO] Focal:    " << focal_px << " px" << std::endl;

    cv::Mat mapL1, mapL2, mapR1, mapR2;
    // Los mapas de remapeo convierten cada frame crudo en una version rectificada fila-con-fila.
    cv::initUndistortRectifyMap(K1, D1, R1, P1, imgSize, CV_32FC1, mapL1, mapL2);
    cv::initUndistortRectifyMap(K2, D2, R2, P2, imgSize, CV_32FC1, mapR1, mapR2);
    d_mapL1.upload(mapL1); d_mapL2.upload(mapL2);
    d_mapR1.upload(mapR1); d_mapR2.upload(mapR2);

    setSGBMParameters(1, 11);
    wls = cv::ximgproc::createDisparityWLSFilter(sgbm);
    wls->setLambda(16000.0);
    wls->setSigmaColor(1.4);

    cv::FileStorage sfs(scale_file, cv::FileStorage::READ);
    if (sfs.isOpened()) {
        // scaleFactor permite corregir el error sistematico final con una calibracion empirica sencilla.
        double sf = 1.0;
        sfs["scaleFactor"] >> sf;
        if (sf >= 0.2 && sf <= 5.0) {
            scaleFactor = sf;
            std::cout << "[INFO] scaleFactor loaded: " << scaleFactor << std::endl;
        } else {
            scaleFactor = 1.0;
            std::cerr << "[WARN] scaleFactor fuera de rango (" << sf
                      << "). Usando 1.0; recalibra con C si hace falta." << std::endl;
        }
        sfs.release();
    }
}

void StereoProcessor::setSGBMParameters(int numDispMult, int blockSz) {
    // Solo permitimos tres configuraciones de disparidad para mantener coste y cobertura controlados.
    int selector = std::clamp(numDispMult, 0, 2);
    int nd = selector == 0 ? 64 : (selector == 1 ? 128 : 192);
    int bs = std::max(3, blockSz % 2 == 0 ? blockSz + 1 : blockSz);

    if (nd != last_nd || bs != last_bs) {
        // P1/P2 regulan suavidad del SGBM: cambios pequenos favorecen continuidad, grandes castigan saltos.
        int p1 = 8 * bs * bs;
        int p2 = 32 * bs * bs;
        auto next = cv::StereoSGBM::create(
            0, nd, bs, p1, p2,
            1, 31, 10, 220, 4,
            cv::StereoSGBM::MODE_SGBM_3WAY);
        sgbm = next;
        rightMatcher = cv::ximgproc::createRightMatcher(sgbm);
        if (wls) {
            // El filtro WLS necesita conocer el matcher izquierdo para usar la geometria correcta.
            wls = cv::ximgproc::createDisparityWLSFilter(sgbm);
            wls->setLambda(16000.0);
            wls->setSigmaColor(1.4);
        }
        last_nd = nd;
        last_bs = bs;
        std::cout << "[INFO] StereoSGBM actualizado: numDisp="
                  << nd << " blockSz=" << bs << std::endl;
    }
}

void StereoProcessor::rectifyImages(const cv::Mat& imgLeft, const cv::Mat& imgRight, cv::Mat& rectL, cv::Mat& rectR) {
    // La rectificacion se hace en GPU para no cargar al CPU principal.
    cv::cuda::GpuMat d_imgL(imgLeft), d_imgR(imgRight);
    cv::cuda::GpuMat d_rectL, d_rectR;
    cv::cuda::remap(d_imgL, d_rectL, d_mapL1, d_mapL2, cv::INTER_LINEAR);
    cv::cuda::remap(d_imgR, d_rectR, d_mapR1, d_mapR2, cv::INTER_LINEAR);
    d_rectL.download(rectL);
    d_rectR.download(rectR);
}

cv::Mat StereoProcessor::getConfidenceMap() const {
    return lastConfidenceMap;
}

void StereoProcessor::computeDisparity(const cv::Mat& rectL, const cv::Mat& rectR, int claheTrack, cv::Mat& dispFilt, cv::Mat& dispF) {
    // SGBM y WLS trabajan sobre intensidad; por eso se parte de gris y no de BGR.
    cv::Mat gL, gR;
    cv::cvtColor(rectL, gL, cv::COLOR_BGR2GRAY);
    cv::cvtColor(rectR, gR, cv::COLOR_BGR2GRAY);

    cv::Mat smoothL, smoothR;
    // El bilateral baja ruido preservando bordes, que son justo lo importante para el matching.
    cv::bilateralFilter(gL, smoothL, 5, 22.0, 5.0);
    cv::bilateralFilter(gR, smoothR, 5, 22.0, 5.0);
    gL = smoothL;
    gR = smoothR;

    if (claheTrack == 1) {
        // CLAHE realza textura local cuando el contraste global es pobre.
        auto clahe = cv::createCLAHE(1.0, cv::Size(16, 16));
        clahe->apply(gL, gL);
        clahe->apply(gR, gR);
    }

    cv::Mat dispL, dispR;
    // dispL = disparidad izquierda, dispR = disparidad derecha usada por el WLS para consistencia izquierda-derecha.
    sgbm->compute(gL, gR, dispL);
    rightMatcher->compute(gR, gL, dispR);
    wls->filter(dispL, gL, dispFilt, dispR);
    lastConfidenceMap = wls->getConfidenceMap();

    if (!dispFilt.empty()) {
        // Los posfiltros finales rematan speckles aislados que aun sobreviven al WLS.
        cv::medianBlur(dispFilt, dispFilt, 3);
        cv::filterSpeckles(dispFilt, 0, 220, 16);
    }
    // OpenCV entrega disparidad fija en 1/16 px; aqui la pasamos a float real en pixeles.
    dispFilt.convertTo(dispF, CV_32F, 1.0 / 16.0);
}
