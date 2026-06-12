#include "YOLODetector.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace {

static cv::Rect clampBox(const cv::Rect& box, const cv::Size& bounds) {
    return box & cv::Rect(0, 0, bounds.width, bounds.height);
}

static Detection makeDetection(float x1, float y1, float x2, float y2, float conf, int classId,
                               float xScale, float yScale, const cv::Size& imageSize) {
    int left = static_cast<int>(std::round(x1 * xScale));
    int top = static_cast<int>(std::round(y1 * yScale));
    int right = static_cast<int>(std::round(x2 * xScale));
    int bottom = static_cast<int>(std::round(y2 * yScale));

    Detection det;
    det.box = clampBox(cv::Rect(left, top, right - left, bottom - top), imageSize);
    det.conf = conf;
    det.classId = classId;
    return det;
}

} // namespace

/**
 * @brief Inicializa ONNX Runtime y carga el modelo de detección facial.
 * @param modelPath Ruta del archivo ONNX exportado.
 * @param inputSize Resolución de entrada usada al crear el blob.
 * @return No devuelve valor.
 * @note Habilita detección facial en tiempo real para ubicar el
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
 * @note Entrega ROIs robustas para el efecto AR y permite
 * excluir rostros de la búsqueda de objetos oscuros/profundidad.
 */
std::vector<Detection> YOLODetector::detect(const cv::Mat& image, float confThreshold, float nmsThreshold) {
    if (image.empty()) return {};

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

    const float xScale = static_cast<float>(image.cols) / inputSize.width;
    const float yScale = static_cast<float>(image.rows) / inputSize.height;

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> classIds;

    auto pushCandidate = [&](float cx, float cy, float w, float h, float conf, int classId) {
        if (conf < confThreshold || w <= 2.0f || h <= 2.0f) return;
        Detection det = makeDetection(cx - w * 0.5f, cy - h * 0.5f, cx + w * 0.5f, cy + h * 0.5f,
                                      conf, classId, xScale, yScale, image.size());
        if (det.box.area() <= 0) return;
        boxes.push_back(det.box);
        scores.push_back(det.conf);
        classIds.push_back(det.classId);
    };

    auto pushCandidateXYXY = [&](float x1, float y1, float x2, float y2, float conf, int classId) {
        if (conf < confThreshold) return;
        Detection det = makeDetection(x1, y1, x2, y2, conf, classId, xScale, yScale, image.size());
        if (det.box.area() <= 0) return;
        boxes.push_back(det.box);
        scores.push_back(det.conf);
        classIds.push_back(det.classId);
    };

    if (shape.size() == 3) {
        const int dim1 = static_cast<int>(shape[1]);
        const int dim2 = static_cast<int>(shape[2]);

        if (dim2 == 6 && dim1 <= 1000) {
            for (int i = 0; i < dim1; ++i) {
                const float* row = data + i * dim2;
                pushCandidateXYXY(row[0], row[1], row[2], row[3], row[4], static_cast<int>(row[5]));
            }
        } else if (dim2 >= 5 && dim2 <= 128) {
            for (int i = 0; i < dim1; ++i) {
                const float* row = data + i * dim2;
                float bestScore = row[4];
                int classId = 0;
                for (int c = 5; c < dim2; ++c) {
                    if (row[c] > bestScore) {
                        bestScore = row[c];
                        classId = c - 5;
                    }
                }
                pushCandidate(row[0], row[1], row[2], row[3], bestScore, classId);
            }
        } else if (dim1 >= 5 && dim1 <= 128) {
            for (int i = 0; i < dim2; ++i) {
                float bestScore = data[4 * dim2 + i];
                int classId = 0;
                for (int c = 5; c < dim1; ++c) {
                    const float score = data[c * dim2 + i];
                    if (score > bestScore) {
                        bestScore = score;
                        classId = c - 5;
                    }
                }
                pushCandidate(data[i], data[dim2 + i], data[2 * dim2 + i], data[3 * dim2 + i],
                              bestScore, classId);
            }
        }
    } else if (shape.size() == 2 && shape[1] >= 6) {
        const int rows = static_cast<int>(shape[0]);
        const int attrs = static_cast<int>(shape[1]);
        for (int i = 0; i < rows; ++i) {
            const float* row = data + i * attrs;
            pushCandidateXYXY(row[0], row[1], row[2], row[3], row[4], static_cast<int>(row[5]));
        }
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, scores, confThreshold, nmsThreshold, keep);
    detections.reserve(keep.size());
    for (int idx : keep) {
        detections.push_back({boxes[idx], scores[idx], classIds[idx]});
    }

    std::sort(detections.begin(), detections.end(), [](const Detection& a, const Detection& b) {
        return a.conf > b.conf;
    });

    return detections;
}
