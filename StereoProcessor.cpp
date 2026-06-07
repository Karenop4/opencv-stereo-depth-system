#include "StereoProcessor.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>

/**
 * @brief Carga calibración estéreo, mapas de rectificación, SGM CUDA y WLS.
 * @param calibration_file YAML con intrínsecos, extrínsecos, proyecciones y Q.
 * @param scale_file YAML con factor de escala fino para la distancia final.
 * @return No devuelve valor.
 * @note Justificación: centraliza la calibración y deja listo el pipeline que
 * convierte disparidad en distancia métrica para la rúbrica de profundidad.
 */
StereoProcessor::StereoProcessor(const std::string& calibration_file, const std::string& scale_file) 
    : scaleFactor(1.0), imgSize(640, 480) {
    
    cv::FileStorage fs(calibration_file, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "[ERROR] No se encontró " << calibration_file << std::endl;
        exit(-1);
    }
    cv::FileNode imageSizeNode = fs["imageSize"];
    if (!imageSizeNode.empty()) {
        imageSizeNode >> imgSize;
        std::cout << "[INFO] imageSize from calibration: " << imgSize.width << "x" << imgSize.height << std::endl;
    }
    cv::Mat K1, K2, D1, D2, R, T, R1, R2, P1, P2, Q;
    fs["K1"]>>K1; fs["D1"]>>D1;
    fs["K2"]>>K2; fs["D2"]>>D2;
    fs["R"] >>R;  fs["T"] >>T;
    fs["R1"]>>R1; fs["R2"]>>R2;
    fs["P1"]>>P1; fs["P2"]>>P2;
    fs["Q"] >>Q;
    fs.release();

    if (R1.empty() || R2.empty() || P1.empty() || P2.empty() || Q.empty()) {
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

    focal_px = std::abs(P1.at<double>(0, 0));
    baseline_mm = 0.0;
    if (!P2.empty() && std::abs(P2.at<double>(0, 0)) > 1e-9) {
        baseline_mm = std::abs(P2.at<double>(0, 3) / P2.at<double>(0, 0));
        std::cout << "[INFO] Baseline from P2: " << baseline_mm << " (same units as calibration)" << std::endl;
    }
    if (baseline_mm <= 0.0 && !Q.empty() && std::abs(Q.at<double>(3, 2)) > 1e-9) {
        baseline_mm = std::abs(1.0 / Q.at<double>(3, 2));
        std::cout << "[INFO] Baseline from Q fallback: " << baseline_mm << " (same units as calibration)" << std::endl;
    }
    if (baseline_mm <= 0.0 && !T.empty() && T.total() >= 3) {
        baseline_mm = cv::norm(T);
        std::cout << "[INFO] Baseline from T fallback: " << baseline_mm << " (same units as calibration)" << std::endl;
    }
    std::cout << "[INFO] Focal:    " << focal_px << " px" << std::endl;

    cv::Mat mapL1, mapL2, mapR1, mapR2;
    cv::initUndistortRectifyMap(K1,D1,R1,P1,imgSize,CV_32FC1,mapL1,mapL2);
    cv::initUndistortRectifyMap(K2,D2,R2,P2,imgSize,CV_32FC1,mapR1,mapR2);
    
    d_mapL1.upload(mapL1); d_mapL2.upload(mapL2);
    d_mapR1.upload(mapR1); d_mapR2.upload(mapR2);

    int initNumDisp = std::max(1, 6) * 16;
    int blockSz = 7;
    int p1 = 8 * 1 * blockSz * blockSz;
    int p2 = 32 * 1 * blockSz * blockSz;
    d_sgm = cv::cuda::createStereoSGM(0, initNumDisp, p1, p2, 5, cv::cuda::StereoSGM::MODE_HH);
    
    wls = cv::ximgproc::createDisparityWLSFilterGeneric(false);
    wls->setLambda(8000.0);
    wls->setSigmaColor(1.5);

    cv::FileStorage sfs(scale_file, cv::FileStorage::READ);
    if (sfs.isOpened()) {
        double sf = 1.0; sfs["scaleFactor"] >> sf;
        if (sf > 0) { scaleFactor = sf; std::cout << "[INFO] scaleFactor loaded: " << scaleFactor << std::endl; }
        sfs.release();
    } else {
        std::cout << "[INFO] scaleFactor default: " << scaleFactor
                  << " (calibracion en mm; usa C solo para ajuste fino)" << std::endl;
    }
}

/**
 * @brief Actualiza los parámetros del matcher SGM CUDA si cambiaron.
 * @param numDispMult Selector de rango de disparidad: 0=64, 1=128, 2=256.
 * @param blockSz Tamaño de bloque solicitado desde la interfaz.
 * @return No devuelve valor.
 * @note Justificación: permite ajustar calidad/ruido durante la demo sin
 * reiniciar la aplicación ni perder la calibración cargada.
 */
void StereoProcessor::setSGBMParameters(int numDispMult, int blockSz) {
    int selector = std::clamp(numDispMult, 0, 2);
    int nd = selector == 0 ? 64 : (selector == 1 ? 128 : 256);
    int bs = std::max(3, blockSz%2==0 ? blockSz+1 : blockSz);
    
    if (nd != last_nd || bs != last_bs) {
        int p1 = 8 * 1 * bs * bs;
        int p2 = 32 * 1 * bs * bs;
        try {
            auto next = cv::cuda::createStereoSGM(0, nd, p1, p2, 5, cv::cuda::StereoSGM::MODE_HH);
            d_sgm = next;
            last_nd = nd;
            last_bs = bs;
            std::cout << "[INFO] StereoSGM actualizado: numDisp=" << nd << " blockSz=" << bs << std::endl;
        } catch (const cv::Exception& e) {
            std::cerr << "[WARN] No se pudo actualizar StereoSGM (se conserva la configuracion anterior): "
                      << e.what() << std::endl;
        }
    }
}

/**
 * @brief Rectifica ambas imágenes usando mapas precalculados en GPU.
 * @param imgLeft Imagen izquierda original.
 * @param imgRight Imagen derecha original.
 * @param rectL Salida rectificada izquierda.
 * @param rectR Salida rectificada derecha.
 * @return No devuelve valor.
 * @note Justificación: alinea epipolares para que la disparidad horizontal sea
 * válida y la reproyección 3D con Q produzca profundidad coherente.
 */
void StereoProcessor::rectifyImages(const cv::Mat& imgLeft, const cv::Mat& imgRight, cv::Mat& rectL, cv::Mat& rectR) {
    cv::cuda::GpuMat d_imgL(imgLeft), d_imgR(imgRight);
    cv::cuda::GpuMat d_rectL, d_rectR;
    
    cv::cuda::remap(d_imgL, d_rectL, d_mapL1, d_mapL2, cv::INTER_LINEAR);
    cv::cuda::remap(d_imgR, d_rectR, d_mapR1, d_mapR2, cv::INTER_LINEAR);
    
    d_rectL.download(rectL);
    d_rectR.download(rectR);
}

/**
 * @brief Calcula mapa de disparidad, aplica WLS y entrega escala float.
 * @param rectL Imagen izquierda rectificada.
 * @param rectR Imagen derecha rectificada.
 * @param claheTrack Activa CLAHE para mejorar contraste local si vale 1.
 * @param dispFilt Salida WLS/mediana en formato de disparidad fijo.
 * @param dispF Salida en CV_32F, dividida entre 16 para cálculos métricos.
 * @return No devuelve valor.
 * @note Justificación: el filtro WLS reduce ruido y conserva bordes, condición
 * crítica para estimar distancias y mostrar un mapa de disparidad de calidad.
 */
void StereoProcessor::computeDisparity(const cv::Mat& rectL, const cv::Mat& rectR, int claheTrack, cv::Mat& dispFilt, cv::Mat& dispF) {
    cv::cuda::GpuMat d_rectL(rectL), d_rectR(rectR);
    cv::cuda::GpuMat d_gL, d_gR;
    
    cv::cuda::cvtColor(d_rectL, d_gL, cv::COLOR_BGR2GRAY);
    cv::cuda::cvtColor(d_rectR, d_gR, cv::COLOR_BGR2GRAY);

    if (claheTrack == 1) {
        auto d_clahe = cv::cuda::createCLAHE(2.0, cv::Size(8, 8));
        d_clahe->apply(d_gL, d_gL);
        d_clahe->apply(d_gR, d_gR);
    }

    cv::cuda::GpuMat d_disp;
    d_sgm->compute(d_gL, d_gR, d_disp);
    
    cv::Mat dispL, gL;
    d_disp.download(dispL);
    d_gL.download(gL);
    
    cv::Mat dispR; 
    wls->filter(dispL, gL, dispFilt, dispR);

    if (!dispFilt.empty()) cv::medianBlur(dispFilt, dispFilt, 5);
    dispFilt.convertTo(dispF, CV_32F, 1.0/16.0);
}
