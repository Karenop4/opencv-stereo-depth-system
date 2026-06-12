#ifndef UTILS_HPP
#define UTILS_HPP

#include <deque>
#include <cmath>

/**
 * @brief Promedia una ventana temporal de mediciones de distancia.
 *
 * @details
 * Mantiene las últimas N lecturas y devuelve su media para amortiguar
 * fluctuaciones frame a frame.
 *
 * @note Esta media móvil estabiliza la distancia mostrada y
 * evita saltos visuales en la lectura de profundidad.
 */
struct Suavizador {
    std::deque<double> hist;
    int maxN;

    /**
     * @brief Construye el suavizador temporal.
     * @param n Número máximo de muestras usadas por la media móvil.
     * @return No devuelve valor.
     * @note Limitar la ventana mantiene baja latencia sin perder
     * estabilidad en la distancia estimada.
     */
    Suavizador(int n = 12) : maxN(n) {}

    /**
     * @brief Agrega una medición y devuelve la media móvil actual.
     * @param val Distancia nueva en centímetros, o valor negativo si no hay lectura.
     * @return Distancia suavizada; -1 si aún no existen muestras válidas.
     * @note Conserva la última lectura válida cuando un frame
     * puntual falla, reduciendo parpadeos en la salida AR.
     */
    double agregar(double val) {
        if (val < 0) return hist.empty() ? -1 : hist.back();
        hist.push_back(val);
        if ((int)hist.size() > maxN) hist.pop_front();
        double s = 0; for (double v : hist) s += v;
        return s / hist.size();
    }
};

/**
 * @brief Filtro de Kalman 1D para la distancia Z en centímetros.
 *
 * @details
 * Modela la distancia como una señal escalar con incertidumbre de proceso y
 * medición configurables.
 *
 * @note Filtra ruido residual de SGM/WLS y mejora la estabilidad
 * métrica requerida para la estimación de profundidad.
 */
struct Kalman1D {
    double x;
    double p;
    double q;
    double r;
    bool iniciado;

    /**
     * @brief Construye el filtro con ruido de proceso y medición.
     * @param processNoise Ruido esperado en la evolución de la distancia.
     * @param measurementNoise Ruido esperado en cada medición estéreo.
     * @return No devuelve valor.
     * @note Permite ajustar respuesta contra suavidad sin tocar
     * la calibración ni el filtro WLS.
     */
    Kalman1D(double processNoise = 0.08, double measurementNoise = 5.5)
        : x(0.0), p(1.0), q(processNoise), r(measurementNoise), iniciado(false) {}

    /**
     * @brief Actualiza el estado del filtro con una nueva medición.
     * @param z Distancia medida en centímetros.
     * @return Distancia filtrada; -1 si todavía no existe estado válido.
     * @note Convierte lecturas ruidosas en una distancia estable
     * para la interfaz y para la calibración manual con la tecla C.
     */
    double actualizar(double z) {
        if (!std::isfinite(z) || z <= 0.0) {
            return iniciado ? x : -1.0;
        }

        if (!iniciado) {
            x = z;
            p = 1.0;
            iniciado = true;
            return x;
        }

        p += q;
        double k = p / (p + r);
        x += k * (z - x);
        p = (1.0 - k) * p;
        return x;
    }
};

#endif // UTILS_HPP
