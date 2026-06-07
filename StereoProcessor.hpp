#ifndef STEREO_PROCESSOR_HPP
#define STEREO_PROCESSOR_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <string>

/**
 * @brief Encapsula la rectificación, cálculo de disparidad y filtrado estéreo.
 *
 * @details
 * Esta clase concentra la tubería que convierte dos imágenes rectificadas en
 * una disparidad filtrada y una profundidad utilizable para AR y calibración.
 */
class StereoProcessor {
public:
    double focal_px;
    double baseline_mm;
    double scaleFactor;
    cv::Size imgSize;
    cv::Mat qMatrix;

    /**
     * @brief Carga la calibración estéreo y la escala de distancia.
     * @param calibration_file Archivo YAML con intrínsecos/extrínsecos y matriz Q.
     * @param scale_file Archivo YAML con el factor de escala fino.
     * @return No devuelve valor.
     * @note Es necesaria para obtener profundidad métrica y cumplir el margen
     * de error solicitado en la rúbrica.
     */
    StereoProcessor(const std::string& calibration_file, const std::string& scale_file);

    /**
     * @brief Ajusta la granularidad de la SGM según el modo seleccionado.
     * @param numDispMult Selector de 64/128/256 disparidades.
     * @param blockSz Tamaño de bloque para el emparejamiento.
     * @return No devuelve valor.
     * @note Es necesario para balancear densidad de mapa, ruido y coste.
     */
    void setSGBMParameters(int numDispMult, int blockSz);
    /**
     * @brief Calcula la disparidad filtrada y su versión en float.
     * @param rectL Imagen rectificada izquierda.
     * @param rectR Imagen rectificada derecha.
     * @param claheTrack Activa o desactiva CLAHE previo.
     * @param dispFilt Salida filtrada en formato original.
     * @param dispF Salida en punto flotante con escala de disparidad.
     * @return No devuelve valor.
     * @note Este paso es crítico para generar un mapa denso con bordes definidos
     * y bajo nivel de ruido.
     */
    void computeDisparity(const cv::Mat& rectL, const cv::Mat& rectR, int claheTrack, cv::Mat& dispFilt, cv::Mat& dispF);
    /**
     * @brief Rectifica el par estéreo usando las mapas precalculadas.
     * @param imgLeft Imagen izquierda original.
     * @param imgRight Imagen derecha original.
     * @param rectL Salida rectificada izquierda.
     * @param rectR Salida rectificada derecha.
     * @return No devuelve valor.
     * @note La rectificación es obligatoria para que la correspondencia sea
     * horizontal y la reproyección 3D sea matemáticamente válida.
     */
    void rectifyImages(const cv::Mat& imgLeft, const cv::Mat& imgRight, cv::Mat& rectL, cv::Mat& rectR);

private:
    cv::cuda::GpuMat d_mapL1, d_mapL2, d_mapR1, d_mapR2;
    cv::Ptr<cv::cuda::StereoSGM> d_sgm;
    cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls;
    
    int last_nd = -1;
    int last_bs = -1;
};

#endif // STEREO_PROCESSOR_HPP
