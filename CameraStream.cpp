#include "CameraStream.hpp"
#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>
#include <sstream>

#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

/**
 * @brief Inicializa el estado compartido de una cámara.
 * @param u URL del stream MJPEG de la ESP32-CAM.
 * @return No devuelve valor.
 * @note Justificación: prepara flags atómicos y timestamp para capturar sin
 * bloquear el procesamiento estéreo principal.
 */
CameraStream::CameraStream(std::string u) : url(u), running(true), connected(false), lastFrameUs(0) {}

/**
 * @brief Solicita la detención del hilo de captura asociado.
 * @param Ninguno.
 * @return No devuelve valor.
 * @note Justificación: permite cerrar recursos de cámara al salir de la demo.
 */
CameraStream::~CameraStream() {
    running = false;
}

/**
 * @brief Captura continuamente frames desde una ESP32-CAM.
 * @param cam Estado compartido de cámara donde se publica el último frame.
 * @return No devuelve valor.
 * @note Justificación: desacopla I/O de red del cálculo estéreo para mantener
 * baja latencia en rectificación, WLS y medición de distancia.
 */
void capture_loop(CameraStream* cam) {
    while (cam->running) {
        cv::VideoCapture cap;
        cap.open(cam->url, cv::CAP_FFMPEG);
        if (!cap.isOpened()) {
            cam->connected = false;
            std::this_thread::sleep_for(std::chrono::seconds(2));
            continue;
        }
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
        cam->connected = true;
        cv::Mat tmp;
        while (cam->running) {
            if (!cap.grab()) break;
            cap.retrieve(tmp);
            if (tmp.empty()) continue;
            std::lock_guard<std::mutex> lock(cam->mtx);
            tmp.copyTo(cam->frame);
            const auto now = std::chrono::steady_clock::now().time_since_epoch();
            cam->lastFrameUs.store(std::chrono::duration_cast<std::chrono::microseconds>(now).count(), std::memory_order_relaxed);
        }
        cap.release();
        cam->connected = false;
    }
}

/**
 * @brief Separa host y puerto desde una URL HTTP.
 * @param url URL de entrada con esquema http://.
 * @param host Salida con el host o IP.
 * @param port Salida con el puerto; usa 80 si no viene especificado.
 * @return true si la URL pudo parsearse.
 * @note Justificación: permite enviar comandos de exposición/ganancia al
 * firmware sin depender de librerías HTTP externas pesadas.
 */
static bool parse_http_url(const std::string& url, std::string& host, std::string& port) {
    const std::string prefix = "http://";
    if (url.rfind(prefix, 0) != 0) return false;

    size_t start = prefix.size();
    size_t slash = url.find('/', start);
    std::string authority = url.substr(start, slash == std::string::npos ? std::string::npos : slash - start);
    size_t colon = authority.find(':');
    if (colon == std::string::npos) {
        host = authority;
        port = "80";
    } else {
        host = authority.substr(0, colon);
        port = authority.substr(colon + 1);
    }
    return !host.empty() && !port.empty();
}

/**
 * @brief Ejecuta una petición HTTP GET mínima contra la ESP32-CAM.
 * @param host Host o IP de la cámara.
 * @param port Puerto HTTP de control.
 * @param path Ruta de control con variable y valor.
 * @return true si la cámara responde con código 200.
 * @note Justificación: fija parámetros ópticos antes de procesar profundidad,
 * reduciendo cambios de iluminación que degradan el mapa de disparidad.
 */
static bool http_get_control(const std::string& host, const std::string& port, const std::string& path) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    addrinfo* result = nullptr;
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &result) != 0) return false;

    int sock = -1;
    for (addrinfo* rp = result; rp != nullptr; rp = rp->ai_next) {
        sock = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (sock == -1) continue;

        timeval timeout{};
        timeout.tv_sec = 1;
        timeout.tv_usec = 500000;
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));

        if (connect(sock, rp->ai_addr, rp->ai_addrlen) == 0) break;
        close(sock);
        sock = -1;
    }
    freeaddrinfo(result);
    if (sock == -1) return false;

    std::ostringstream req;
    req << "GET " << path << " HTTP/1.1\r\n"
        << "Host: " << host << "\r\n"
        << "Connection: close\r\n\r\n";
    std::string request = req.str();

    bool ok = send(sock, request.c_str(), request.size(), 0) == static_cast<ssize_t>(request.size());
    if (ok) {
        char buffer[128];
        ssize_t n = recv(sock, buffer, sizeof(buffer) - 1, 0);
        if (n > 0) {
            buffer[n] = '\0';
            ok = std::strstr(buffer, "200") != nullptr;
        } else {
            ok = false;
        }
    }
    close(sock);
    return ok;
}

/**
 * @brief Configura variables del firmware de una ESP32-CAM.
 * @param streamUrl URL del stream usado para deducir host de control.
 * @param controls Lista de pares variable/valor.
 * @return true si todos los comandos recibieron respuesta HTTP 200.
 * @note Justificación: estabiliza exposición, ganancia y reloj de cámara antes
 * de estimar profundidad, mejorando consistencia entre vistas izquierda/derecha.
 */
bool configure_esp32_cam(const std::string& streamUrl, const std::vector<std::pair<std::string, int>>& controls) {
    std::string host, port;
    if (!parse_http_url(streamUrl, host, port)) return false;

    bool allOk = true;
    for (const auto& [var, val] : controls) {
        std::ostringstream path;
        path << "/control?var=" << var << "&val=" << val;
        bool ok = http_get_control(host, "80", path.str());
        allOk = allOk && ok;
        std::cout << "[ESP32] " << host << " " << var << "=" << val
                  << (ok ? " OK" : " sin respuesta") << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(80));
    }

    if (port != "81") {
        std::cout << "[ESP32] Stream usando puerto " << port
                  << "; controles enviados al puerto HTTP 80." << std::endl;
    }
    return allOk;
}
