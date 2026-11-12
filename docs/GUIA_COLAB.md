# Guía para correr el notebook final en Google Colab

**Tiempo total:** unos 75–100 minutos con GPU T4, más ~5 minutos de preparación.
**Necesitan:** una cuenta de Google y el archivo `store-sales-time-series-forecasting.zip` (22 MB; es el que
ya tienen en Descargas).

## 1. Abrir el notebook

Abran este enlace (abre el notebook directamente desde GitHub):

https://colab.research.google.com/github/johansoriano63-ux/bread-dairy-forecasting/blob/main/notebooks/Final_Experiments_Bread_Dairy.ipynb

Si Colab dice que no puede abrirlo, el repositorio todavía es privado o no se ha subido. En ese caso,
descarguen `notebooks/Final_Experiments_Bread_Dairy.ipynb` de GitHub y súbanlo en Colab con
**Archivo → Subir notebook**.

Luego hagan **Archivo → Guardar una copia en Drive**, para que sus cambios no se pierdan.

## 2. Activar la GPU

**Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU → Guardar.**

> Sin GPU el notebook también corre, pero tarda más de 12 horas.

## 3. Subir los datos

1. En la barra izquierda, hagan clic en el ícono de **carpeta** (Archivos).
2. Arrastren `store-sales-time-series-forecasting.zip` a esa ventana. Queda en `/content/`.
   Esperen a que termine la barra de carga (abajo a la izquierda).
3. No hace falta descomprimirlo: el notebook lo descomprime solo.

> Los archivos subidos así se borran cuando Colab se desconecta. Si eso pasa, súbanlo de nuevo.

## 4. Correr todo

**Entorno de ejecución → Ejecutar todas** (`Ctrl + F9`).

* Al inicio, Colab pide **permiso para acceder a Google Drive**. Acéptenlo con la cuenta de uno de ustedes.
  Los resultados se van guardando en `Mi unidad/freshcast_results`.
* Si al instalar paquetes Colab dice **"Reiniciar sesión"**, acepten y vuelvan a hacer *Ejecutar todas*.

Qué esperar en cada parte (tiempos aproximados con T4):

| Sección del notebook | Qué hace | Tiempo |
|---|---|---|
| 0–4 | Instalación, datos, covariables, análisis exploratorio | 2–3 min |
| 5 | Modelo estacional ingenuo y ETS | ~1 min |
| 8 | Ajuste de hiperparámetros de la LSTM (5 configuraciones) | 8–12 min |
| 9 | 5 semillas × (LSTM, GRU, LightGBM) | 45–60 min |
| 10–12 | Recalibración conformal, tablas, figuras, pedidos, alertas | 1–2 min |
| 13 | Gradientes integrados y ablación (6 reentrenamientos) | 8–10 min |
| 14 | Temporal Fusion Transformer | 10–25 min |
| 15 | Pronóstico 16–31 ago y exportación para la app | ~1 min |

Mientras corre, la sección 9 imprime una línea por modelo, por ejemplo
`LSTM seed 42 {...}: best epoch 27, val loss 0.00540, 3.1 min`. Si ven esas líneas, todo va bien.

## 5. Si Colab se desconecta

Pasa si la pestaña queda inactiva mucho rato o si se acaba la cuota gratuita de GPU.

1. Vuelvan a conectar, revisen que siga en **T4 GPU** y suban otra vez el `.zip` (paso 3).
2. **Ejecutar todas** otra vez. Los modelos que ya terminaron se cargan desde Drive (`freshcast_results/cache`)
   y el notebook sigue desde donde iba.

Para evitar desconexiones, dejen la pestaña de Colab abierta y el computador sin suspenderse.
Si se acaba la cuota de GPU, otro integrante puede seguir con su cuenta: compartan la carpeta
`freshcast_results` de Drive y cópienla en su propia `Mi unidad`.

## 6. Al terminar

* La última celda descarga **`results_final.zip`** (también queda una copia en `Mi unidad/freshcast_results`).
* Déjenlo en **Descargas** y avísenme. Con ese archivo actualizo el paper, la app y el repositorio.

## Problemas frecuentes

| Mensaje | Qué hacer |
|---|---|
| `AssertionError: Put these files in /content/data: [...]` | No se subió el `.zip` o no terminó de cargar. Repitan el paso 3. |
| `device: cpu` en la sección 7 | No está activa la GPU. Repitan el paso 2 y vuelvan a ejecutar todo. |
| `CUDA out of memory` | Entorno de ejecución → Reiniciar sesión → Ejecutar todas (retoma desde la caché). |
| `TFT could not run: ...` | El resto de resultados sigue siendo válido. Copien el mensaje y envíenmelo. |
| Pide autorizar Drive otra vez | Normal después de reconectar; acepten. |
| `credential propagation was unsuccessful` / `mount failed` | La autorización de Drive no se completó. Corran la celda otra vez y en la ventana elijan **la misma cuenta del Colab** y marquen **todas** las casillas de permisos. Si sigue fallando, el notebook continúa sin Drive (aviso `WARNING: Google Drive could not be mounted`) y guarda todo en `/content/results`. |
| No quieren usar Drive | En la celda de configuración cambien `SAVE_TO_DRIVE = True` por `False` (si se desconecta, empieza de cero). |
