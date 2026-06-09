# INFORME TÉCNICO: SISTEMA INTEGRADOR DE VISIÓN ESTÉREO Y ESTIMACIÓN DE PROFUNDIDAD

Carrera de Computación - Asignatura: Visión Artificial

Universidad Politécnica Salesiana

## 1. Introducción

El desarrollo de sistemas de percepción tridimensional es un componente central en robótica, automatización e interacción hombre-máquina. La visión estéreo permite estimar profundidad a partir de dos vistas capturadas desde posiciones ligeramente separadas, reproduciendo de forma aproximada el principio de percepción binocular humana. No obstante, cuando esta técnica se implementa sobre hardware de bajo costo, aparecen limitaciones prácticas como latencia de red, desincronización entre cámaras, variaciones de iluminación y ruido en la correspondencia de pixeles.

Este proyecto implementa un sistema de visión estéreo en C++ con OpenCV para capturar dos transmisiones MJPEG provenientes de ESP32-CAM, rectificarlas, calcular disparidad, estimar distancia en centimetros y superponer un efecto de realidad aumentada sobre la imagen de color. La solución busca mantener una lectura estable en tiempo real y ofrecer una arquitectura modular para pruebas, calibración y defensa técnica.

## 2. Planteamiento del problema

La medición de profundidad con sensores de bajo costo presenta tres dificultades principales. La primera es la asincronía temporal: al recibir dos flujos de video por red, cada cámara puede entregar frames en instantes diferentes, lo que degrada la geometría estéreo. La segunda es la inestabilidad lumínica: los automatismos del sensor, como autoexposición, ganancia automática y balance de blancos, pueden variar de manera independiente en cada cámara y afectar la correspondencia entre vistas. La tercera es la presencia de ruido en la disparidad: texturas homogéneas, bordes poco definidos y oclusiones generan regiones imprecisas o vacías en el mapa de profundidad.

En consecuencia, el problema no es solo calcular disparidad, sino estabilizar toda la cadena de adquisición, rectificación, estimación y visualización para obtener una distancia utilizable en tiempo real.

## 3. Propuesta metodológica

### 3.1 Arquitectura de hardware y configuración de cámara

El sistema utiliza dos ESP32-CAM como fuentes de video y las configura desde el lado software mediante peticiones HTTP. En el código se desactivan controles automáticos del sensor y se fijan parámetros consistentes para ambas cámaras, con el objetivo de reducir variaciones entre la vista izquierda y la derecha. Esta decisión mejora la similitud radiométrica y favorece el proceso de correspondencia estéreo.

La estación de trabajo actúa como un entorno de procesamiento centralizado, mientras que la captura queda distribuida en los dos dispositivos embebidos. Esta separación permite utilizar cámaras económicas sin depender de hardware especializado de adquisición estéreo.

### 3.2 Arquitectura concurrente de software

La captura de cada cámara se ejecuta en un hilo independiente mediante `std::thread`. Cada hilo actualiza el ultimo frame disponible dentro de un estado compartido protegido por `std::mutex`, mientras que banderas atomicas indican conexión, ejecución y tiempo de llegada del ultimo frame. Esta organización reduce la latencia percibida por el hilo principal, que siempre trabaja con el frame más reciente sin bloquearse por I/O de red.

Adicionalmente, el sistema mide la edad temporal de cada frame para descartar imágenes viejas o desfasadas. Con ello se evita que la etapa de disparidad procese dos capturas tomadas en instantes muy distintos, lo cual introduciría errores geométricos.

### 3.3 Pipeline de procesamiento estéreo

El flujo de visión sigue las etapas siguientes:

1. Captura concurrente de ambas transmisiones ESP32-CAM.
2. Validación de sincronía temporal mediante la edad de los frames.
3. Redimensionamiento al tamaño de calibración.
4. Corrección opcional de balance de blancos por software.
5. Rectificación estéreo usando la calibración almacenada en `parametros_stereo.yml`.
6. Cálculo de disparidad con StereoSGBM.
7. Filtrado de la disparidad con WLS y limpieza adicional con operaciones morfológicas.
8. Reproyección a 3D mediante la matriz Q.
9. Estimación de distancia en centimetros para el objeto central.
10. Estabilización temporal de la distancia mediante suavizado y filtro de Kalman 1D.
11. Visualización del mapa de disparidad y de la escena con superposición de información.

### 3.4 Rectificación y disparidad

La rectificación alinea las líneas epipolares para que la búsqueda de correspondencias se reduzca a desplazamientos horizontales. El proyecto carga las matrices de calibración y, a partir de ellas, genera mapas de remapeo para rectificar ambas imágenes. Esta etapa es esencial porque el cálculo de disparidad presupone que los puntos correspondientes se encuentran sobre la misma línea horizontal.

La disparidad se calcula con StereoSGBM, un algoritmo de coincidencia semi-global que ofrece un compromiso adecuado entre densidad de reconstrucción y costo computacional. Sobre el resultado crudo se aplica filtrado WLS, que suaviza áreas ruidosas sin destruir bordes relevantes. Después, se utiliza filtrado adicional y limpieza de componentes pequeñas para mejorar la visualización y la robustez de la estimación.

### 3.5 Estimación de profundidad

La profundidad se obtiene a partir de la relación geométrica entre focal, baseline y disparidad. El sistema usa la matriz Q de reproyección para convertir la imagen de disparidad en coordenadas tridimensionales y extraer la componente Z. Además, para calibraciones manuales se dispone de un factor de escala que permite ajustar la lectura a una distancia real conocida.

Para estabilizar la salida, la medición pasa por una media movil y luego por un filtro de Kalman 1D. Este tratamiento reduce oscilaciones espurias provocadas por ruido de captura, cambios leves de textura y variaciones pequeñas en la disparidad estimada.

### 3.6 Capa de realidad aumentada

El sistema también integra una capa visual adicional sobre la imagen de color. Primero detecta rostros con un modelo ONNX ejecutado mediante YOLO/ONNX Runtime. Luego, si existen dos rostros válidos, la clase `FaceSwapper` usa landmarks faciales de dlib para intercambiar las caras mediante transformaciones afines, corrección de color y composición por máscara.

Este módulo no altera el cálculo de profundidad, pero demuestra que el pipeline puede incorporar procesamiento visual de alto nivel sin romper la estructura general del sistema.

## 4. Análisis cuantitativo y resultados

### 4.1 Precisión de la distancia

Para validar la exactitud del sistema, se recomienda medir objetos a distancias conocidas y comparar la distancia real con la distancia estimada por el sistema. El error porcentual puede expresarse como:

Error (%) = (|Distancia real - Distancia medida| / Distancia real) x 100

Tabla sugerida para el informe final:

| Distancia real | Distancia medida | Error absoluto | Error porcentual |
| --- | --- | --- | --- |
| 30 cm |  |  |  |
| 60 cm |  |  |  |
| 90 cm |  |  |  |

Si ya dispones de mediciones experimentales, puedes colocar aquí los valores obtenidos en laboratorio. En caso contrario, conviene no inventar cifras y dejar esta tabla como espacio de validación experimental.

### 4.2 Rendimiento visual

El sistema fue diseñado para operar en tiempo real y mostrar simultáneamente la vista color y el mapa de disparidad. En la práctica, el rendimiento depende de la resolución de captura, la estabilidad de red, el tamaño de bloque de SGBM, la activación de CLAHE y el peso de los módulos de detección facial. Por ello, el informe puede describir el rendimiento como una tasa de refresco fluida y estable bajo condiciones controladas, sin fijar un número si todavía no ha sido medido de forma reproducible.

### 4.3 Estabilización temporal

La combinación de promedio móvil y Kalman 1D reduce el parpadeo de la distancia mostrada en pantalla. Esta estabilización es especialmente útil cuando el objeto está casi inmóvil y la lectura debe permanecer consistente para una demostración o una aplicación interactiva.

## 5. Restricciones técnicas y aprendizajes

Uno de los aprendizajes principales del proyecto es que la calibración geométrica y la estabilidad radiométrica deben mantenerse coherentes en todo el sistema. Si la cámara cambia su exposición o balance de blancos de forma independiente, la correspondencia estéreo pierde calidad. Del mismo modo, si la transmisión por red introduce demasiada latencia o desincronización, la estimación de profundidad se vuelve inestable.

También se concluye que no todo el pipeline debe sobrecargarse con afirmaciones de aceleración por GPU. En este proyecto, la rectificación se apoya en OpenCV con soporte CUDA cuando está disponible, pero la disparidad principal y el filtrado WLS dependen de los módulos de OpenCV correspondientes y de la configuración del entorno. Esta precisión en la descripción técnica fortalece el informe frente a preguntas de defensa.

## 6. Conclusiones

1. Se implementó un sistema de visión estéreo funcional en C++ que integra captura concurrente, rectificación, disparidad, reproyección 3D y visualización en tiempo real.
2. La arquitectura multihilo mejora la respuesta frente a la latencia de red y permite utilizar el frame más reciente de cada cámara sin bloquear el procesamiento principal.
3. El uso de StereoSGBM, filtrado WLS y reproyección con la matriz Q permite estimar profundidad en centimetros con una salida suficientemente estable para demostraciones y pruebas experimentales.
4. La media movil y el filtro de Kalman 1D reducen fluctuaciones temporales de la lectura, mejorando la presentación de la distancia en la interfaz.
5. La integración de detección facial y FaceSwap demuestra que el sistema soporta capas adicionales de realidad aumentada sin interferir con la estimación estéreo.

## 7. Referencia rápida del proyecto

- Captura concurrente: `CameraStream.cpp`
- Flujo principal e interfaz: `main_stereo.cpp`
- Rectificación, disparidad y filtrado: `StereoProcessor.cpp`
- Detección facial: `YOLODetector.cpp`
- Intercambio facial: `FaceSwapper.cpp`
- Parámetros de calibración: `parametros_stereo.yml`
- Factor de escala: `scale.yml`

## 8. Guion corto para defensa

El sistema usa dos ESP32-CAM para capturar una escena desde dos puntos de vista. Cada cámara se lee en un hilo independiente para reducir la latencia de red. Luego rectificamos las imágenes con la calibración, calculamos la disparidad con StereoSGBM, filtramos el mapa con WLS y reproyectamos a 3D usando la matriz Q. La distancia del objeto central se estabiliza con media móvil y Kalman 1D. Además, el proyecto integra un módulo de realidad aumentada que detecta rostros con YOLO y permite intercambiar caras con dlib.
