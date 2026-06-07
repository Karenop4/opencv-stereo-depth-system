#include "YOLODetector.hpp"
#include <iostream>

/**
 * @brief Inicializa ONNX Runtime y carga el modelo de detección facial.
 * @param modelPath Ruta del archivo ONNX exportado.
 * @param inputSize Resolución de entrada usada al crear el blob.
 * @return No devuelve valor.
 * @note Justificación: habilita detección facial en tiempo real para ubicar el
 * efecto AR sin interferir con la estimación estéreo de profundidad.
 */
YOLODetector::YOLODetector(const std::string& modelPath, const cv::Size& inputSize)
    : inputSize(inputSize) {
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    session = std::make_unique<Ort::Session>(env, modelPath.c_str(), sessionOptions);

    Ort::AllocatorWithDefaultOptions allocator;

    size_t numInputNodes = session->GetInputCount();
    for (size_t i = 0; i < numInputNodes; i++) {
        Ort::AllocatedStringPtr name = session->GetInputNameAllocated(i, allocator);
        inputNamesAllocated.push_back(std::move(name));
        inputNames.push_back(inputNamesAllocated.back().get());
    }

    size_t numOutputNodes = session->GetOutputCount();
    for (size_t i = 0; i < numOutputNodes; i++) {
        Ort::AllocatedStringPtr name = session->GetOutputNameAllocated(i, allocator);
        outputNamesAllocated.push_back(std::move(name));
        outputNames.push_back(outputNamesAllocated.back().get());
    }
}

/**
 * @brief Ejecuta inferencia y transforma cajas a coordenadas de la imagen.
 * @param image Frame BGR de entrada.
 * @param confThreshold Confianza mínima aceptada.
 * @return Lista de detecciones faciales aceptadas.
 * @note Justificación: entrega ROIs robustas para el efecto AR y permite
 * excluir rostros de la búsqueda de objetos oscuros/profundidad.
 */
std::vector<Detection> YOLODetector::detect(const cv::Mat& image, float confThreshold) {
    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 1.0 / 255.0, inputSize, cv::Scalar(), true, false);

    std::vector<int64_t> inputDims = {1, 3, inputSize.height, inputSize.width};
    
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, (float*)blob.data, blob.total(), inputDims.data(), inputDims.size());

    auto outputTensors = session->Run(Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor, 1, outputNames.data(), outputNames.size());

    std::vector<Detection> detections;
    if (outputTensors.empty() || !outputTensors.front().IsTensor()) return detections;

    float* data = outputTensors.front().GetTensorMutableData<float>();
    auto type_info = outputTensors.front().GetTensorTypeAndShapeInfo();
    auto shape = type_info.GetShape();

    float x_scale = static_cast<float>(image.cols) / inputSize.width;
    float y_scale = static_cast<float>(image.rows) / inputSize.height;

    if (shape.size() == 3 && shape[2] == 6) {
        int num_boxes = shape[1];
        for (int i = 0; i < num_boxes; i++) {
            float x1 = data[i * 6 + 0];
            float y1 = data[i * 6 + 1];
            float x2 = data[i * 6 + 2];
            float y2 = data[i * 6 + 3];
            float conf = data[i * 6 + 4];
            int classId = static_cast<int>(data[i * 6 + 5]);

            if (conf >= confThreshold) {
                int left = static_cast<int>(x1 * x_scale);
                int top = static_cast<int>(y1 * y_scale);
                int right = static_cast<int>(x2 * x_scale);
                int bottom = static_cast<int>(y2 * y_scale);

                Detection det;
                det.box = cv::Rect(left, top, right - left, bottom - top);
                det.conf = conf;
                det.classId = classId;
                detections.push_back(det);
            }
        }
    }
    return detections;
}
