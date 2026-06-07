#ifndef CAMERA_STREAM_HPP
#define CAMERA_STREAM_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <mutex>
#include <atomic>
#include <vector>

/**
 * @brief Estado compartido de una cámara ESP32-CAM capturada desde un hilo.
 *
 * @details
 * Agrupa URL, frame más reciente, sincronización y estado de conexión para
 * reducir latencia de captura y evitar bloquear el hilo principal de visión.
 */
struct CameraStream {
    std::string url;
    cv::Mat frame;
    std::mutex mtx;
    std::atomic<bool> running;
    std::atomic<bool> connected;
    std::atomic<long long> lastFrameUs;

    /**
     * @brief Construye un stream de cámara asociado a una URL HTTP.
     * @param u URL del stream MJPEG de la cámara.
     * @return No devuelve valor.
     * @note Es necesario para desacoplar la captura concurrente del
     * procesamiento estéreo y disminuir jitter.
     */
    CameraStream(std::string u);

    /**
     * @brief Detiene la captura asociada al stream.
     * @param Ninguno.
     * @return No devuelve valor.
     * @note Permite cerrar limpiamente los hilos al salir de la aplicación.
     */
    ~CameraStream();
    
    /**
     * @brief Deshabilita copias del estado compartido de cámara.
     * @param Ninguno.
     * @return No devuelve valor.
     * @note Justificación: evita duplicar mutex, atomics y buffers de imagen
     * usados por los hilos de captura.
     */
    CameraStream(const CameraStream&) = delete;

    /**
     * @brief Deshabilita asignación del estado compartido de cámara.
     * @param Ninguno.
     * @return Referencia al stream asignado, aunque esta operación está eliminada.
     * @note Justificación: mantiene un único dueño lógico por cámara y previene
     * condiciones de carrera accidentales.
     */
    CameraStream& operator=(const CameraStream&) = delete;
};

/**
 * @brief Bucle de captura continuo para una cámara.
 * @param cam Puntero al estado compartido de la cámara.
 * @return No devuelve valor.
 * @note Es clave para reducir latencia y mantener el último frame disponible
 * sin bloquear el pipeline principal de estimación de profundidad.
 */
void capture_loop(CameraStream* cam);

/**
 * @brief Envía parámetros de control al firmware de una ESP32-CAM.
 * @param streamUrl URL base de la cámara.
 * @param controls Lista de pares variable/valor a configurar.
 * @return true si todos los controles se aplicaron; false en caso contrario.
 * @note Justifica el ajuste de AEC, AGC, contraste y XCLK para reducir
 * saturación, variación de exposición y jitter visual entre frames.
 */
bool configure_esp32_cam(const std::string& streamUrl, const std::vector<std::pair<std::string, int>>& controls);

#endif // CAMERA_STREAM_HPP
