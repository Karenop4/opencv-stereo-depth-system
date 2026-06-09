# Guia de Sustentacion

## Idea General

El proyecto toma dos imagenes de camaras separadas horizontalmente, busca la diferencia horizontal entre ambas vistas, convierte esa diferencia en profundidad y luego usa esa profundidad para medir la distancia del objeto central.

Formula mental:

```text
Z = f * B / d
```

- `f`: focal en pixeles.
- `B`: distancia entre camaras o baseline.
- `d`: disparidad en pixeles.
- `Z`: profundidad.

## Que Hace Cada Archivo

- [main_stereo.cpp](/home/user/Documentos/7mo%20ciclo/Vision_Computador/Proyecto/progC/main_stereo.cpp): une todo; captura, rectifica, calcula disparidad, mide distancia, dibuja la interfaz y maneja teclado.
- [CameraStream.cpp](/home/user/Documentos/7mo%20ciclo/Vision_Computador/Proyecto/progC/CameraStream.cpp): habla con las ESP32-CAM y mantiene el ultimo frame actualizado.
- [StereoProcessor.cpp](/home/user/Documentos/7mo%20ciclo/Vision_Computador/Proyecto/progC/StereoProcessor.cpp): parte matematica estereo pesada.
- [DepthUtils.cpp](/home/user/Documentos/7mo%20ciclo/Vision_Computador/Proyecto/progC/DepthUtils.cpp): limpieza de ruido, ROI central, medianas, percentiles y distancia robusta.
- [FaceSwapper.cpp](/home/user/Documentos/7mo%20ciclo/Vision_Computador/Proyecto/progC/FaceSwapper.cpp): efecto AR.
- [YOLODetector.cpp](/home/user/Documentos/7mo%20ciclo/Vision_Computador/Proyecto/progC/YOLODetector.cpp): deteccion facial para alimentar el FaceSwap.

## Variables que Te Pueden Preguntar

### En `main_stereo.cpp`

- `brilloTrack`: offset de brillo aplicado solo a la rama de profundidad.
- `contTrack`: contraste multiplicativo.
- `claheTrack`: activa CLAHE antes del matching estereo.
- `wbSoftTrack`: corrige el tinte verde por software.
- `numDispAuto`: selector de `64`, `128` o `192` disparidades.
- `blockSz`: tamano de bloque de `StereoSGBM`.
- `modoCercaTrack`: fuerza rango de disparidad alto para objetos cercanos.
- `dispTemporalF`: suavizado temporal del mapa de disparidad.
- `lastAcceptedDistanceCm`: ultima distancia considerada valida.
- `rejectedDistanceCount`: contador de lecturas descartadas por saltos imposibles.

### En `StereoProcessor.cpp`

- `focal_px`: focal de la camara rectificada.
- `baseline_mm`: separacion fisica entre camaras.
- `qMatrix`: matriz de reproyeccion 3D.
- `scaleFactor`: correccion empirica final en centimetros.
- `lastConfidenceMap`: mapa de confianza del WLS.

## Pipeline del `main`

1. Carga calibracion y escala.
2. Configura firmware de ambas ESP32-CAM.
3. Lanza dos hilos de captura.
4. Espera a tener un frame valido de cada camara.
5. Comprueba que los frames no esten demasiado viejos o desincronizados.
6. Ajusta tamano al de la calibracion.
7. Corrige balance de blancos si se activa.
8. Rectifica ambas vistas.
9. Ajusta brillo y contraste para la rama de profundidad.
10. Elige el rango de disparidad.
11. Ejecuta `StereoSGBM`.
12. Filtra con `WLS`.
13. Reproyecta con `Q`.
14. Crea mascaras de confianza y limpia ruido.
15. Busca el objeto central o usa fallbacks.
16. Calcula distancia robusta.
17. Suaviza con media movil y Kalman.
18. Dibuja la interfaz y revisa teclado.

## Preguntas

**Por que desactivar AEC, AGC y AWB?**

Porque si cada camara ajusta exposicion, ganancia o color por su cuenta, ambas vistas dejan de parecerse y el matching de disparidad empeora.

**Por que usar CLAHE?**

Porque resalta textura local en superficies pobres; eso ayuda a encontrar correspondencias donde el contraste global no alcanza.

**Por que usar WLS despues de SGBM?**

Porque SGBM solo no basta en escenas ruidosas. WLS rellena huecos y suaviza respetando bordes.

**Por que usar mediana dentro de un ROI?**

Porque la mediana resiste outliers; unos pocos pixeles malos no arruinan toda la lectura.

**Por que la ventana de disparidad a veces tenia una franja negra a la izquierda?**

Porque `StereoSGBM` pierde columnas al inicio cuando el rango de disparidad es alto. En la visualizacion actual esa banda invalida se recorta solo para mostrar mejor el mapa.

## Sobre Q

`Q` es la matriz que convierte `(x, y, d, 1)` a coordenadas 3D. Luego OpenCV divide por `W` y entrega `(X, Y, Z)`. En este proyecto la distancia mostrada es `Z`, no `X` ni `Y`.

## Los Hilos

Cada ESP32-CAM se captura en un hilo distinto para que la latencia de una no bloquee a la otra. El hilo principal solo toma el ultimo frame listo y sigue procesando.

## Estabilidad

No se muestra la distancia cruda. Primero se calcula una lectura robusta en el centro, luego se rechazan saltos improbables y finalmente se suaviza con media movil y Kalman.

