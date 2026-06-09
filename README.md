# Proyecto Integrador: Vision Estereo con ESP32-CAM, Profundidad y AR

## Resumen

Este proyecto implementa un sistema de vision estereo en C++ con OpenCV que:

- captura dos streams MJPEG desde dos ESP32-CAM en paralelo;
- rectifica ambas vistas con una calibracion estereo;
- calcula un mapa de disparidad denso con `StereoSGBM`;
- filtra la disparidad con `WLS` y filtros espaciales;
- reproyecta la disparidad a 3D con la matriz `Q`;
- estima la distancia del objeto central en centimetros;
- estabiliza la lectura con media movil y Kalman 1D;
- aplica un efecto AR de intercambio facial sobre la imagen izquierda.

El ejecutable principal es `main_stereo`.

## Archivos Principales

- [main_stereo.cpp](/home/user/Documentos/7mo%20ciclo/Vision_Computador/Proyecto/progC/main_stereo.cpp): orquestacion completa del pipeline.
- [CameraStream.cpp](/home/user/Documentos/7mo%20ciclo/Vision_Computador/Proyecto/progC/CameraStream.cpp): captura concurrente y configuracion HTTP de las ESP32-CAM.
- [StereoProcessor.cpp](/home/user/Documentos/7mo%20ciclo/Vision_Computador/Proyecto/progC/StereoProcessor.cpp): rectificacion, SGBM, WLS y conversion a disparidad float.
- [DepthUtils.cpp](/home/user/Documentos/7mo%20ciclo/Vision_Computador/Proyecto/progC/DepthUtils.cpp): utilidades de confianza, ROI central, estadistica robusta y distancia.
- [FaceSwapper.cpp](/home/user/Documentos/7mo%20ciclo/Vision_Computador/Proyecto/progC/FaceSwapper.cpp): efecto AR con landmarks de dlib.
- [YOLODetector.cpp](/home/user/Documentos/7mo%20ciclo/Vision_Computador/Proyecto/progC/YOLODetector.cpp): deteccion facial con ONNX Runtime.
- [GUIA_SUSTENTACION.md](/home/user/Documentos/7mo%20ciclo/Vision_Computador/Proyecto/progC/GUIA_SUSTENTACION.md): guia corta para repasar antes de exponer.

## Compilacion

```bash
make
```

El `Makefile` compila actualmente:

```make
SRCS = main_stereo.cpp CameraStream.cpp StereoProcessor.cpp DepthUtils.cpp FaceSwapper.cpp YOLODetector.cpp
```

## Ejecucion

Con URLs por defecto:

```bash
./main_stereo
```

Con URLs manuales:

```bash
./main_stereo http://IP_IZQ:81/stream http://IP_DER:81/stream
```

## Flujo del Sistema

1. `main_stereo.cpp` crea `StereoProcessor`, `YOLODetector` y `FaceSwapper`.
2. Se configuran ambas ESP32-CAM por HTTP con AEC, AGC y AWB desactivados.
3. Se levanta un hilo por camara mediante `capture_loop()`.
4. El hilo principal toma el ultimo frame disponible de cada stream.
5. Si los frames estan demasiado viejos o desincronizados, se descartan.
6. Las imagenes se ajustan al tamano calibrado y se rectifican.
7. `StereoProcessor::computeDisparity()` calcula `dispF` con `StereoSGBM`.
8. El resultado se filtra con `WLS`, mediana y filtros adicionales.
9. `cv::reprojectImageTo3D()` usa `qMatrix` para generar `points3D`.
10. `DepthUtils` extrae una distancia robusta en la ROI central.
11. La distancia pasa por media movil, Kalman y `scaleFactor`.
12. Se dibujan la vista izquierda, el mapa de disparidad y el FaceSwap.

## Controles en la Interfaz

- `Brillo`: desplaza intensidad de la rama de profundidad.
- `Contraste x10`: ganancia de contraste aplicada antes de la disparidad.
- `CLAHE`: activa o desactiva ecualizacion adaptativa previa al matching.
- `WB software`: compensa el tinte verde si AWB esta apagado en firmware.
- `Area min x100`: area minima usada en detectores auxiliares.
- `Block Size`: tamano de bloque de `StereoSGBM`.
- `Dist max cm`: umbral de profundidad para el fallback por nube 3D.
- `AR YOLO`: activa el pipeline de deteccion facial y FaceSwap.
- `Grabar`: guarda un AVI con imagen izquierda + disparidad.
- `Modo cerca`: fuerza el uso de `192` disparidades para objetos cercanos.

Teclas:

- `C`: calibra `scaleFactor` con una distancia real conocida.
- `S`: guarda la vista izquierda y el mapa de disparidad actual.
- `ESC`: sale del programa.

## Evidencia Frente a la Rubrica

### 1. Arquitectura de captura y hardware

- Hay un hilo por camara en `capture_loop()`.
- Cada hilo publica el ultimo frame, su timestamp y un estimado de FPS.
- `configure_esp32_cam()` fija controles del sensor antes del pipeline.
- En `ESP32_CONTROLES` se documenta `aec=0`, `agc=0`, `awb=0` y `xclk`.

### 2. Calidad del mapa de disparidad

- El algoritmo base es `cv::StereoSGBM::create(...)`.
- Se aplica CLAHE opcional en `StereoProcessor::computeDisparity()`.
- Se usa `cv::ximgproc::createDisparityWLSFilter(...)`.
- La visualizacion usa percentiles, bilateral, cierre morfologico y CLAHE visual.

### 3. Precision y estabilidad de la distancia

- `cv::reprojectImageTo3D(dispF, points3D, stereoProc.qMatrix, true)` usa explicitamente `Q`.
- `DepthUtils::distanciaRobustaCm()` combina:
  - `Z` de la nube 3D reproyectada;
  - la formula estereo `Z = f * B / d`.
- La distancia final mostrada se suaviza con `Suavizador` y `Kalman1D`.
- `scaleFactor` permite calibracion metrica fina.

### 4. Efecto AR

- `YOLODetector` detecta rostros con ONNX.
- `FaceSwapper` usa landmarks de dlib para el intercambio facial.
- El resultado se dibuja sobre la imagen izquierda a color.

## Respuestas Cortas para Sustentacion

**Por que dos hilos?**

Para desacoplar la red del procesamiento; cada ESP32-CAM puede retrasarse sin bloquear la otra ni congelar el pipeline estereo.

**Por que mutex y atomics?**

El mutex protege el `frame` mientras se copia. Los atomics permiten leer `running`, `connected`, `lastFrameUs` y `fps` sin bloqueo pesado.

**Que hace la matriz Q?**

Convierte coordenadas de imagen y disparidad en coordenadas 3D. De ahi se toma `Z` como profundidad real.

**Que hace WLS?**

Suaviza el mapa de disparidad guiandose por los bordes de la imagen izquierda; reduce ruido sin borrar contornos importantes.

**Por que no basta con `Z = f * B / d`?**

Porque la disparidad puntual puede tener outliers. Por eso se combinan estadisticas robustas, mascara de confianza y la nube 3D reproyectada.

**Por que media movil y Kalman?**

La media movil amortigua ruido rapido. El Kalman agrega memoria e incertidumbre para estabilizar sin volver la lectura demasiado lenta.

## Dependencias de Runtime

- `parametros_stereo.yml`
- `scale.yml`
- `shape_predictor_68_face_landmarks.dat`
- `yolov26/runs/detect/yolov26_faces/weights/best.onnx`
- ONNX Runtime en `onnxruntime-linux-x64-1.18.0/lib`

## Nota Practica

Antes de sustentar, conviene repasar [GUIA_SUSTENTACION.md](/home/user/Documentos/7mo%20ciclo/Vision_Computador/Proyecto/progC/GUIA_SUSTENTACION.md), porque resume el papel de cada archivo, las variables que suelen preguntar y un guion corto del flujo completo.
