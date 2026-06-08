#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <memory>

/**
 * @brief Resultado de una detección con caja, confianza y clase.
 *
 * @note Justificación: encapsula la salida mínima del detector para ubicar
 * rostros y evitar que el AR invada regiones usadas por profundidad.
 */
struct Detection {
    /** @brief Caja del objeto detectado en coordenadas de imagen. */
    cv::Rect box;
    /** @brief Confianza de detección normalizada entre 0 y 1. */
    float conf;
    /** @brief Clase predicha por el modelo ONNX. */
    int classId;
};

/**
 * @brief Wrapper de inferencia ONNX para detectar rostros en tiempo real.
 *
 * @details
 * Se usa para sostener la capa AR sobre regiones faciales y evitar que el
 * efecto interfiera con la profundidad principal.
 */
class YOLODetector {
public:
    /**
     * @brief Carga el modelo ONNX y prepara la sesión de inferencia.
     * @param modelPath Ruta del modelo exportado.
     * @param inputSize Tamaño de entrada esperado por el modelo.
     * @return No devuelve valor.
     * @note Necesario para mantener detección facial en tiempo real dentro del
     * flujo de AR sin detener la tubería estéreo.
     */
    YOLODetector(const std::string& modelPath, const cv::Size& inputSize = cv::Size(640, 640));
    /**
     * @brief Ejecuta la detección sobre una imagen.
     * @param image Imagen de entrada en BGR.
     * @param confThreshold Umbral mínimo de confianza para aceptar cajas.
     * @return Vector de detecciones filtradas.
     * @note Es necesaria para localizar caras de forma estable antes de aplicar
     * el efecto AR y no confundir la estimación de profundidad.
     */
    std::vector<Detection> detect(const cv::Mat& image, float confThreshold = 0.5f, float nmsThreshold = 0.45f);

private:
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "YOLODetector"};
    Ort::SessionOptions sessionOptions;
    std::unique_ptr<Ort::Session> session;
    cv::Size inputSize;

    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;
    
    std::vector<Ort::AllocatedStringPtr> inputNamesAllocated;
    std::vector<Ort::AllocatedStringPtr> outputNamesAllocated;
};
