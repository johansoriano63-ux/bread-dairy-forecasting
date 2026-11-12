# FreshCast – guion del video promocional (3:00)

Formato sugerido: grabación de pantalla de la app (OBS o la grabadora de Windows `Win + Alt + R`) con la voz en off
de los tres integrantes y 2–3 tomas cortas de ustedes a cámara. Resolución 1080p. Subirlo a YouTube como
**No listado** y pegar el enlace en el paper (sección III-J) y en el README.

La narración está en español; si el curso lo pide en inglés, se traduce tal cual (las cifras entre corchetes se
llenan con los resultados finales).

| Tiempo | Imagen | Narración (quién) |
|---|---|---|
| 0:00–0:15 | Toma de pan y leche en un supermercado (foto o video libre de derechos, p. ej. Pexels). Texto: *"Cada día, una tienda decide cuánto pan y cuánta leche tener."* | **Stephany:** "Si pide de más, el pan se bota al final del día. Si pide de menos, el cliente se va sin comprar. Y esa decisión se toma todos los días en cada tienda." |
| 0:15–0:35 | Logo de FreshCast + título del proyecto + Universidad de Ibagué. | **Stephany:** "Somos Stephany, Johan y Cristian, de Sistemas Inteligentes en la Universidad de Ibagué. Construimos FreshCast, un agente inteligente que pronostica la demanda de panadería y lácteos y recomienda cuánto pedir." |
| 0:35–1:00 | Gráfica de ventas por día de la semana (Fig. EDA) y mapa/lista de las 54 tiendas. | **Johan:** "Usamos datos reales de 54 supermercados de Corporación Favorita en Ecuador: casi cinco años de ventas diarias, con promociones, festivos, días de quincena y precio del petróleo. El domingo se vende un 30 % más que el promedio, y las promociones mueven mucho más a los lácteos que al pan." |
| 1:00–1:30 | Diagrama de arquitectura (Fig. 2 del paper), animado por bloques. | **Johan:** "El núcleo es una red neuronal recurrente LSTM tipo codificador–decodificador: lee ocho semanas de historia y, con lo que ya se sabe del futuro, como promociones y festivos, predice los próximos 16 días. No da un solo número: da un rango, del 10 al 90 %, y lo recalibramos para que de verdad acierte el 80 % de las veces." |
| 1:30–2:20 | **Demo de la app** (grabación de pantalla): 1) elegir tienda 44 y *Bread*; 2) mover el control de costo de faltante 1→4 y ver cómo sube el pedido; 3) cambiar a *Dairy* y mostrar el stock que se arrastra; 4) pestaña *Why?*: "la promoción sube el pronóstico [X] %"; 5) pestaña *Alerts* en modo backtest. | **Cristian:** "Así se ve para el encargado de la tienda. Elijo la tienda y la familia y veo el pronóstico con su rango. Si quedarme sin pan me cuesta cuatro veces más que botarlo, muevo este control y el agente sube el pedido. En lácteos, que duran varios días, solo pide lo que falta para completar el nivel objetivo. En *¿Por qué?* explica qué factores mueven el pronóstico de cada día, y en *Alertas* avisa cuando las ventas reales se salen del rango esperado." |
| 2:20–2:45 | Tabla de resultados (Tabla de test del paper) y figura de cobertura antes/después. | **Stephany:** "Comparamos la red con cinco alternativas, con cinco semillas y una ventana de prueba que solo se usó una vez. La LSTM reduce el error frente a los modelos estadísticos en [X] % y [compite con / supera a] LightGBM; con la regla de pedido el costo simulado baja hasta [X] % en pan y la regla con inventario reduce el desperdicio en lácteos." |
| 2:45–3:00 | Pantalla final: enlaces a GitHub, paper y app; nombres y correos; logo de la universidad. | **Todos:** "FreshCast: menos desperdicio, menos faltantes, decisiones que se entienden. El código, los datos y el informe están en GitHub." |

## Lista de tomas para grabar la demo (app en `streamlit run app/app.py`)

1. Periodo *Forecast · Aug 16–31*, familia *Bread / bakery*, tienda *44 · Quito*. Mostrar las tarjetas de arriba.
2. Mover `c_u : c_o` de 1 a 4 lentamente: la línea roja punteada sube y cambia el pedido del primer día.
3. Cambiar a *Dairy*, vida útil 7 días: mostrar la columna *stock carried* de la tabla.
4. Pestaña *Why?*: mover el control de día hasta uno con promoción o quincena y leer la frase.
5. Periodo *Backtest*, pestaña *Forecast & order*: triángulos naranja/rojo de alerta sobre las ventas reales.
6. Pestaña *Alerts (all stores)* y *Model quality*.

## Consejos

* Graben la voz aparte (celular cerca de la boca, cuarto silencioso) y móntenla sobre la pantalla en CapCut,
  Clipchamp (viene con Windows) o DaVinci Resolve.
* Mantengan el cursor quieto entre acciones y hagan zoom (Clipchamp → *Zoom*) en las cifras importantes.
* Música de fondo libre de derechos a volumen bajo (biblioteca de audio de YouTube).
* Revisen que dure **3:00 o menos**.
