# Proyecto Integrador: Vision Estereo con ESP32-CAM, Profundidad y AR

## Objetivo

Este proyecto implementa un sistema de vision estereo en C++ con OpenCV para capturar dos streams ESP32-CAM, rectificarlos, calcular un mapa de disparidad denso, estimar distancia real en centimetros y superponer un efecto de realidad aumentada dinamico sobre la imagen a color.

El ejecutable principal es `main_stereo`, construido desde:

```make
SRCS = main_stereo.cpp CameraStream.cpp StereoProcessor.cpp FaceSwapper.cpp YOLODetector.cpp
```

## Checklist de Rubrica

| Criterio | Estado | Evidencia en el codigo |
| --- | --- | --- |
| Arquitectura de captura y hardware | Cumplido | `CameraStream` usa hilos para capturar ambas ESP32-CAM; `configure_esp32_cam()` ajusta AEC, AGC, brillo, contraste y XCLK. |
| Calidad del mapa de disparidad | Cumplido | `StereoProcessor::computeDisparity()` usa CLAHE opcional, CUDA StereoSGM, filtro WLS, mediana y visualizacion con percentiles. |
| Precision y estabilidad de distancia | Cumplido | `cv::reprojectImageTo3D()` usa `qMatrix`; tambien se valida con formula `Z = f * B / d`; la lectura pasa por media movil y Kalman 1D. |
| Efecto AR | Cumplido | `FaceSwapper` realiza intercambio facial; `dibujarEfectoARDistancia()` cambia escala, color y comportamiento segun la distancia medida. |
| Defensa individual | Preparado | Este README resume que responder sobre Q, threads, locks, SGM, WLS, Kalman y calibracion. |

## Flujo General del Sistema

1. Se cargan calibracion y escala desde `parametros_stereo.yml` y `scale.yml`.
2. Se configuran las ESP32-CAM con parametros de firmware.
3. Se crean dos hilos, uno por camara, para capturar en paralelo.
4. El hilo principal copia el ultimo frame valido de cada camara usando mutex.
5. Se valida sincronizacion temporal comparando la edad de ambos frames.
6. Las imagenes se redimensionan al tamano calibrado y se rectifican.
7. Se calcula la disparidad con StereoSGM en GPU.
8. Se filtra la disparidad con WLS y mediana para reducir ruido.
9. Se reproyecta la disparidad a 3D con la matriz Q.
10. Se detecta un objeto, se estima distancia y se estabiliza con media movil y Kalman.
11. Se dibuja el efecto AR dinamico y el mapa de disparidad.

## 1. Arquitectura de Captura y Hardware

Archivos clave:

- `CameraStream.hpp`
- `CameraStream.cpp`
- `main_stereo.cpp`

Cada camara se representa con:

- `url`: direccion del stream MJPEG.
- `frame`: ultimo frame recibido.
- `mtx`: mutex para proteger el acceso al frame.
- `running`: bandera atomica para detener el hilo.
- `connected`: bandera atomica de conexion.
- `lastFrameUs`: timestamp monotono del ultimo frame.

La funcion `capture_loop(CameraStream* cam)` corre en un hilo independiente por camara. Abre el stream con `cv::VideoCapture`, reduce el buffer con `CAP_PROP_BUFFERSIZE = 1`, captura frames y actualiza el ultimo frame bajo un `std::lock_guard<std::mutex>`.

Esto reduce latencia porque el hilo principal no espera a que llegue un frame por red; siempre usa el ultimo frame disponible.

La configuracion de firmware se hace con `configure_esp32_cam()`:

- `aec`: auto-exposicion.
- `aec2`: ajuste adicional de exposicion.
- `agc`: auto-ganancia.
- `gainceiling`: limite de ganancia.
- `brightness` y `contrast`: ajuste base de imagen.
- `xclk`: reloj estable para la camara.

Respuesta oral sugerida:

> Uso un hilo por camara para desacoplar red y procesamiento. El mutex protege el frame compartido, y los atomics permiten leer estados como `running`, `connected` y timestamp sin bloquear. Esto baja jitter porque el pipeline de disparidad no queda esperando I/O.

## 2. Calidad del Mapa de Disparidad

Archivo clave:

- `StereoProcessor.cpp`

La clase `StereoProcessor` concentra la parte estereo:

- Carga matrices intrinsecas y extrinsecas.
- Calcula o carga rectificacion.
- Crea mapas de remapeo con `cv::initUndistortRectifyMap`.
- Sube mapas a GPU.
- Usa `cv::cuda::StereoSGM`.
- Aplica filtro `cv::ximgproc::DisparityWLSFilter`.

Preprocesamiento:

- Se convierte BGR a gris.
- Si `CLAHE` esta activo, se aplica `cv::cuda::createCLAHE(2.0, Size(8, 8))`.
- CLAHE mejora contraste local y ayuda cuando la iluminacion no es uniforme.

Algoritmo de disparidad:

- Se usa StereoSGM en CUDA.
- `numDisp` puede ser 64, 128 o 256.
- `blockSz` se ajusta desde trackbar y se fuerza a impar.
- `P1 = 8 * blockSize^2`.
- `P2 = 32 * blockSize^2`.

Posprocesamiento:

- WLS suaviza zonas ruidosas conservando bordes.
- `medianBlur` reduce speckles.
- En la visualizacion se usan percentiles para evitar que outliers dominen el contraste.
- Tambien se aplica filtro bilateral, cierre morfologico y CLAHE visual.

Respuesta oral sugerida:

> SGM calcula correspondencias densas buscando la mejor disparidad por pixel. CLAHE mejora textura antes del matching. WLS funciona como posfiltro guiado por la imagen izquierda: suaviza regiones planas, pero respeta discontinuidades de borde, por eso reduce ruido sin borrar contornos.

## 3. Precision y Estabilidad de la Distancia

Archivos clave:

- `main_stereo.cpp`
- `utils.hpp`
- `StereoProcessor.cpp`

La profundidad se obtiene por dos rutas:

1. Reproyeccion 3D:

```cpp
cv::reprojectImageTo3D(dispF, points3D, stereoProc.qMatrix, true);
```

Luego se extrae Z desde `points3D` y se convierte de milimetros a centimetros.

2. Formula estereo directa:

```cpp
Z = (focal_px * baseline_mm) / disparidad
```

En el codigo:

```cpp
zFromDispCm = (stereoProc.focal_px * stereoProc.baseline_mm) / dispMed / 10.0;
```

Se comparan ambas rutas. Si difieren demasiado, el sistema prioriza la medicion por disparidad para evitar una lectura incoherente.

Estabilizacion:

- `Suavizador`: media movil de 16 muestras.
- `Kalman1D`: filtro de Kalman escalar.

Media movil:

```cpp
promedio = suma(historial) / cantidad
```

Kalman 1D:

```cpp
p = p + q
k = p / (p + r)
x = x + k * (z - x)
p = (1 - k) * p
```

Donde:

- `x`: distancia estimada.
- `z`: nueva medicion.
- `p`: incertidumbre actual.
- `q`: ruido de proceso.
- `r`: ruido de medicion.
- `k`: ganancia de Kalman.

Tambien existe calibracion manual con tecla `C`. Si se conoce la distancia real, se calcula:

```cpp
scaleFactor = distancia_real_cm / distancia_medida_raw_cm
```

Respuesta oral sugerida:

> La matriz Q transforma coordenadas de imagen y disparidad en coordenadas 3D. Para la distancia uso Z. Como la disparidad tiene ruido, tomo medianas dentro del ROI, rechazo valores invalidos, aplico media movil y finalmente Kalman 1D para que la lectura no fluctue mas de lo permitido.

## 4. Efecto de Realidad Aumentada

Archivos clave:

- `FaceSwapper.hpp`
- `FaceSwapper.cpp`
- `YOLODetector.cpp`
- `main_stereo.cpp`

El AR tiene dos componentes:

1. Deteccion facial con YOLO ONNX.
2. Intercambio facial con dlib landmarks.
3. Overlay dinamico de distancia sobre el objeto detectado.

`YOLODetector` carga `best.onnx` y devuelve cajas faciales. Si hay dos rostros, `FaceSwapper::swapFaces()`:

- Obtiene landmarks de 68 puntos.
- Calcula transformaciones afines.
- Crea mascaras faciales.
- Warpea cada cara hacia la otra.
- Corrige color con especificacion de histograma.
- Suaviza bordes con erosion y blur.
- Compone el resultado sobre el frame.

El efecto que reacciona a distancia esta en `dibujarEfectoARDistancia()`:

- Si el objeto esta cerca, aumenta el halo.
- Cambia color segun profundidad.
- Cambia la barra de proximidad.
- Se dibuja sobre la transmision a color.

Respuesta oral sugerida:

> El FaceSwap es el efecto visual original. Para cumplir la condicion dinamica de la rubrica, tambien dibujo un reticulo AR cuyo tamano, color y barra de proximidad dependen de la distancia filtrada. Asi el AR no es estatico: responde directamente a la profundidad medida.

## 5. Matriz Q: Explicacion para Defensa

La matriz Q viene de la calibracion estereo y permite convertir:

```text
(x, y, disparity, 1) -> (X, Y, Z, W)
```

Despues OpenCV normaliza por `W` y devuelve puntos 3D.

Idea central:

- Si la disparidad es grande, el objeto esta cerca.
- Si la disparidad es pequena, el objeto esta lejos.
- La profundidad depende de focal, baseline y disparidad.

Formula conceptual:

```text
Z = f * B / d
```

Donde:

- `f`: distancia focal en pixeles.
- `B`: baseline entre camaras.
- `d`: disparidad.
- `Z`: profundidad.

## 6. Preguntas Probables y Respuestas Cortas

**Por que se usan dos hilos?**

Para capturar ambas camaras en paralelo y evitar que la latencia de red bloquee el calculo de disparidad.

**Por que se usa mutex?**

Porque el hilo de captura escribe `frame` mientras el hilo principal lo lee. El mutex evita leer un frame a medio copiar.

**Por que `lastFrameUs` es atomico?**

Porque se lee desde el hilo principal y se escribe desde el hilo de captura sin necesitar bloquear.

**Que hace CLAHE?**

Mejora el contraste local. Ayuda a que SGM encuentre correspondencias en zonas con poca textura o iluminacion desigual.

**Que hace WLS?**

Filtra la disparidad usando informacion de bordes de la imagen guia. Reduce ruido y conserva contornos.

**Por que se usa mediana en el ROI?**

Porque la mediana es robusta ante outliers. Unos pocos pixeles malos no cambian drasticamente la distancia.

**Por que Kalman y media movil?**

La media movil amortigua ruido rapido. Kalman modela incertidumbre y produce una estimacion estable sin retrasar demasiado la respuesta.

**Que pasa si hay frames desincronizados?**

Se descartan si la edad supera 250 ms o si la diferencia entre camaras supera 90 ms, evitando disparidades falsas.

**Por que se usa `scaleFactor`?**

Para hacer un ajuste fino entre la medicion calculada y una distancia real conocida durante calibracion manual.

**Que archivos son runtime obligatorio?**

- `main_stereo`
- `parametros_stereo.yml`
- `scale.yml`
- `shape_predictor_68_face_landmarks.dat`
- `yolov26/runs/detect/yolov26_faces/weights/best.onnx`
- `onnxruntime-linux-x64-1.18.0/lib/libonnxruntime.so.1.18.0`

## 7. Archivos que No Entran al Ejecutable Principal

Estos archivos pueden mantenerse como soporte, pero no forman parte de `main_stereo`:

- `depth_map.cpp`: herramienta de calibracion, util si se necesita recalibrar.
- `test_cuda.cpp`: prueba aislada de CUDA.
- `inspect_onnx.py`: inspeccion del modelo.
- `yolov26/train_faces.py`: entrenamiento.
- `yolov26/export_onnx.py`: exportacion a ONNX.
- `yolov26/auto_annotate.py`: apoyo para anotacion.
- `.pt` y datasets: entrenamiento, no runtime.

No deben agregarse al `SRCS` del Makefile final.

## 8. Como Compilar y Ejecutar

Compilar:

```bash
make -B
```

Ejecutar:

```bash
./main_stereo
```

Controles:

- `Brillo`: compensacion visual.
- `Contraste x10`: ganancia de contraste.
- `CLAHE`: activa o desactiva preprocesamiento.
- `Area min x100`: area minima del objeto.
- `Num Disp`: 64, 128 o 256 disparidades.
- `Block Size`: tamano de bloque SGM.
- `Dist max cm`: distancia maxima para deteccion por profundidad.
- `C`: calibracion manual de distancia.
- `S`: guardar frame de vision y mapa de profundidad.
- `ESC`: salir.

## 9. Guion Corto para Presentacion

> Nuestro sistema usa dos ESP32-CAM configuradas desde C++ para estabilizar exposicion, ganancia y reloj. Cada camara se lee en un hilo independiente, y el hilo principal procesa el ultimo frame sincronizado. Despues rectificamos con la calibracion estereo, calculamos disparidad con StereoSGM en CUDA, aplicamos CLAHE y WLS para mejorar densidad y reducir ruido, y reproyectamos a 3D con la matriz Q. La distancia se obtiene de Z y se valida tambien con la formula f por baseline dividido para disparidad. Para estabilizar la lectura usamos media movil y Kalman 1D. Finalmente, integramos un efecto AR original: intercambio facial y un reticulo dinamico que cambia con la distancia medida.

