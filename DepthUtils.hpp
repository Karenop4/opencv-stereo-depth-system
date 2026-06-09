#pragma once

#include <opencv2/opencv.hpp>

namespace depth {

bool rangoDisparidadPercentil(const cv::Mat& dispF, const cv::Mat& validMask,
                              double pLow, double pHigh, double& low, double& high);
cv::Mat mascaraConfianzaDisparidad(const cv::Mat& dispF, const cv::Mat& validMask);
cv::Mat mascaraPrimerPlanoCentral(const cv::Mat& dispF, const cv::Rect& roi, const cv::Mat& validMask);
cv::Mat filtrarComponentesPequenos(const cv::Mat& mask, int minArea);
cv::Rect cajaDesdeMascara(const cv::Mat& mask, int minArea);
float medianaDisparidadConMascara(const cv::Mat& dispF, const cv::Rect& roi, const cv::Mat& mask);
float percentilDisparidadConMascara(const cv::Mat& dispF, const cv::Rect& roi, const cv::Mat& mask, double percentile);
double medianaZValida(const cv::Mat& points3D, const cv::Rect& roi);
cv::Rect roiCentralMedicion(const cv::Size& size);
int elegirModoDisparidadAuto(double distanciaEstableCm, int modoActual);
bool medicionDistanciaPlausible(double medicionCm, double referenciaCm);
double distanciaRobustaCm(const cv::Mat& dispF, const cv::Mat& points3D, const cv::Rect& roi,
                          const cv::Mat& validDispMask, double focalPx, double baselineMm);
bool detectarObjetoPorProfundidad(const cv::Mat& points3D, double depthThreshCm, double areaMin,
                                  cv::Rect& bestBox, double& meanDepthCm);

} // namespace depth
