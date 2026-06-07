#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>

/**
 * @brief Realiza intercambio facial con corrección geométrica y de color.
 *
 * @details
 * Se usa como efecto AR complementario para aportar una capa visual original
 * y reactiva sobre la escena capturada.
 */
class FaceSwapper {
public:
    /**
     * @brief Carga el predictor de landmarks de 68 puntos.
     * @param landmarks_path Ruta del archivo de landmarks de dlib.
     * @return No devuelve valor.
     * @note Necesario para estimar la geometría facial con suficiente estabilidad.
     */
    explicit FaceSwapper(const std::string& landmarks_path);

    /**
     * @brief Libera recursos del intercambiador facial.
     * @param Ninguno.
     * @return No devuelve valor.
     * @note Justificación: mantiene una interfaz explícita para liberar el
     * predictor de landmarks al finalizar el efecto AR.
     */
    ~FaceSwapper() = default;

    /**
     * @brief Intercambia dos rostros dentro del mismo frame.
     * @param frame Frame destino en el que se compone el efecto.
     * @param rect_ann Caja del primer rostro.
     * @param rect_bob Caja del segundo rostro.
     * @return No devuelve valor.
     * @note Es la pieza visual que hace que la capa AR sea distinta y visible
     * en tiempo real sobre la escena.
     */
    void swapFaces(cv::Mat& frame, const cv::Rect& rect_ann, const cv::Rect& rect_bob);

private:
    dlib::shape_predictor pose_model;

    cv::Mat small_frame;
    cv::Size frame_size;
    cv::Rect rect_ann, rect_bob;
    cv::Rect big_rect_ann, big_rect_bob;

    dlib::cv_image<dlib::bgr_pixel> dlib_frame;
    dlib::rectangle dlib_rects[2];
    dlib::full_object_detection shapes[2];

    cv::Point points_ann[9], points_bob[9];
    cv::Point2f affine_transform_keypoints_ann[3], affine_transform_keypoints_bob[3];
    cv::Size feather_amount;

    cv::Mat trans_ann_to_bob, trans_bob_to_ann;
    cv::Mat mask_ann, mask_bob;
    cv::Mat warpped_mask_ann, warpped_mask_bob;
    cv::Mat refined_ann_and_bob_warpped, refined_bob_and_ann_warpped;
    cv::Mat refined_masks;

    cv::Mat face_ann, face_bob;
    cv::Mat warpped_face_ann, warpped_face_bob;
    cv::Mat warpped_faces;

    int source_hist_int[3][256];
    int target_hist_int[3][256];
    float source_histogram[3][256];
    float target_histogram[3][256];
    uint8_t LUT[3][256];

    /**
     * @brief Prepara el recorte mínimo que contiene ambos rostros.
     * @param frame Frame completo de la cámara izquierda.
     * @param rect_ann Caja del primer rostro en coordenadas del frame completo.
     * @param rect_bob Caja del segundo rostro en coordenadas del frame completo.
     * @return Vista del frame recortada alrededor de ambos rostros.
     * @note Justificación: reduce el área procesada por dlib/OpenCV y baja la
     * latencia del efecto AR en hardware limitado.
     */
    cv::Mat getMinFrame(const cv::Mat& frame, const cv::Rect& rect_ann, const cv::Rect& rect_bob);

    /**
     * @brief Extrae landmarks y puntos geométricos de ambos rostros.
     * @param frame Recorte donde se encuentran los dos rostros.
     * @return No devuelve valor.
     * @note Justificación: los landmarks permiten alinear el intercambio sin
     * romper la coherencia espacial de la capa AR.
     */
    void getFacePoints(const cv::Mat& frame);

    /**
     * @brief Calcula las transformaciones afines entre ambos rostros.
     * @param Ninguno; usa los puntos faciales almacenados.
     * @return No devuelve valor.
     * @note Justificación: alinear geometría facial evita deformaciones bruscas
     * en el efecto AR.
     */
    void getTransformationMatrices();

    /**
     * @brief Genera máscaras base para cada rostro.
     * @param Ninguno; usa los puntos faciales almacenados.
     * @return No devuelve valor.
     * @note Justificación: limita el efecto AR a la región facial y evita
     * contaminar el resto de la escena.
     */
    void getMasks();

    /**
     * @brief Deforma las máscaras hacia la geometría destino.
     * @param Ninguno; usa las matrices afines almacenadas.
     * @return No devuelve valor.
     * @note Justificación: conserva la forma de la cara destino para que el
     * overlay AR sea visualmente consistente.
     */
    void getWarppedMasks();

    /**
     * @brief Combina máscaras originales y deformadas.
     * @param Ninguno; usa las máscaras internas ya calculadas.
     * @return Máscara refinada de las zonas donde se pegarán los rostros.
     * @note Justificación: reduce bordes falsos y mejora la calidad visual del AR.
     */
    cv::Mat getRefinedMasks();

    /**
     * @brief Extrae las regiones faciales desde el recorte de trabajo.
     * @param Ninguno; usa el frame y las máscaras internas.
     * @return No devuelve valor.
     * @note Justificación: separa contenido facial de fondo para una composición limpia.
     */
    void extractFaces();

    /**
     * @brief Deforma los rostros hacia sus posiciones destino.
     * @param Ninguno; usa las matrices afines y caras extraídas.
     * @return Imagen con los rostros deformados sobre fondo negro.
     * @note Justificación: produce el contenido final que será mezclado con el frame.
     */
    cv::Mat getWarppedFaces();

    /**
     * @brief Corrige el color de los rostros deformados.
     * @param Ninguno; usa las regiones faciales internas.
     * @return No devuelve valor.
     * @note Justificación: iguala iluminación para que el AR no se vea como un parche.
     */
    void colorCorrectFaces();

    /**
     * @brief Suaviza los bordes de una máscara facial.
     * @param refined_masks Región de máscara a erosionar y difuminar.
     * @return No devuelve valor.
     * @note Justificación: el suavizado oculta costuras y mejora la calidad del overlay.
     */
    void featherMask(cv::Mat& refined_masks);

    /**
     * @brief Mezcla los rostros procesados dentro del recorte final.
     * @param Ninguno; usa el frame, rostros y máscara internos.
     * @return No devuelve valor.
     * @note Justificación: compone la salida AR visible para la evaluación.
     */
    void pasteFacesOnFrame();

    /**
     * @brief Aplica especificación de histograma entre dos regiones.
     * @param source_image Región fuente que define la distribución de color.
     * @param target_image Región destino que será recoloreada.
     * @param mask Máscara de píxeles válidos para la corrección.
     * @return No devuelve valor.
     * @note Justificación: reduce diferencias de color entre cámaras/rostros y
     * mejora la naturalidad del efecto AR.
     */
    void specifyHistogram(const cv::Mat& source_image, cv::Mat target_image, const cv::Mat& mask);
};
