# Beer Tap Counter

Una aplicacion que cuenta las cervezas servidas desde un dispensador de cerveza de doble grifo analizando grabaciones de video. El sistema observa las grabaciones del area de los grifos de un bar, identifica eventos individuales de servido y asigna cada uno al **Grifo A** o al **Grifo B**. Los resultados se persisten en una base de datos y se muestran a traves de una interfaz web.

El objetivo es proporcionar a los duenos de bares un conteo automatizado y preciso de cervezas a partir de sus camaras de seguridad o monitoreo existentes, sin requerir ninguna modificacion de hardware en los grifos.

Construido como caso de estudio para **Intern Full Stack & AI Developer** en Gambooza.

---

## Arquitectura

```
Streamlit (frontend)  ──HTTP──>  FastAPI (backend + pipeline ML)  ──>  SQLite
     :8501                              :8000                        app.db
```

- **Backend:** API REST con FastAPI. Acepta subidas de video, ejecuta el procesamiento ML en segundo plano y expone los conteos de cervezas y eventos de servido a traves de endpoints REST.
- **Frontend:** Aplicacion Streamlit. Proporciona una interfaz para subir videos, configurar las regiones ROI (donde estan los grifos en el cuadro), ver conteos por grifo y explorar una linea temporal de eventos de servido.
- **Pipeline ML:** Un pipeline de vision por computadora en multiples etapas que combina YOLO-World (deteccion de objetos zero-shot), BoT-SORT (seguimiento multi-objeto), re-enlace de tracks y SAM3 (segmentacion de manijas de grifo). Un pre-filtro rapido opcional basado en diferencia de pixeles maneja videos largos eficientemente.
- **Base de datos:** SQLite con ORM SQLAlchemy. Almacena metadatos de video y eventos individuales de servido.

---

## Configuracion del Entorno

### Prerequisitos

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** gestor de paquetes (maneja entornos virtuales y resolucion de dependencias)
- **GPU NVIDIA + CUDA** (requerido para inferencia de YOLO y SAM3)
- **Pesos de modelos ML** colocados en `data/models/`:
  - `yolov8x-worldv2.pt` (modelo YOLO-World)
  - `sam3.pt` (Segment Anything Model 3)

### Instalar Dependencias

```bash
# Clonar el repositorio
git clone <repo-url>
cd gambooza_case_study

# Instalar todas las dependencias (crea .venv automaticamente)
uv sync
```

Esto instala todas las dependencias de produccion y desarrollo definidas en `pyproject.toml`, incluyendo PyTorch, Ultralytics, FastAPI, Streamlit y herramientas de testing.

---

## Ejecucion Local

Se necesitan dos terminales, una para el backend y otra para el frontend.

**Terminal 1 -- Iniciar el backend:**

```bash
uv run uvicorn backend.main:app --port 8000 --reload
```

El backend se inicia en `http://localhost:8000`. En la primera ejecucion crea la base de datos SQLite en `data/db_files/app.db`.

**Terminal 2 -- Iniciar el frontend:**

```bash
cd frontend
uv run streamlit run app.py
```

El frontend se inicia en `http://localhost:8501`.

**Verificar que todo funciona:**

1. Abrir `http://localhost:8501` en el navegador.
2. Seleccionar un nombre de restaurante y un ID de camara (o crear nuevos).
3. Subir un video (.mp4 o .mov).
4. Si no existe una configuracion ROI para esa combinacion restaurante+camara, el asistente ROI te guiara para seleccionar las regiones de los grifos.
5. El procesamiento inicia automaticamente. Refrescar para verificar el progreso.

### Ejecutar Tests

```bash
uv run pytest tests/ -v    # 59 tests
```

### Linting y Formateo

```bash
uv run ruff check .        # lint (agregar --fix para corregir automaticamente)
uv run ruff format .       # formatear
```

Los hooks de pre-commit estan disponibles para ejecutar esto automaticamente en cada commit:

```bash
uv run pre-commit install
```

---

## Ejecucion con Docker

Docker Compose ejecuta tanto el backend como el frontend en contenedores, con acceso a GPU para inferencia ML.

### Prerequisitos

- **Docker** y **Docker Compose**
- **nvidia-container-toolkit** instalado y configurado (para acceso a GPU dentro de los contenedores)
- Una GPU NVIDIA con soporte CUDA

### Paso a paso

```bash
# 1. Crear los directorios de datos necesarios en el host
mkdir -p data/{db_files,models,roi_configs,results}

# 2. Colocar los pesos de los modelos en data/models/
#    - yolov8x-worldv2.pt
#    - sam3.pt

# 3. Construir e iniciar ambos servicios
docker compose up --build
```

- Frontend: `http://localhost:8501`
- API Backend: `http://localhost:8000`

Todos los datos persisten en el directorio `./data/` del host, que se monta como volumen en el contenedor del backend en `/app/mount_data`. Esto significa que la base de datos, videos subidos, configuraciones ROI y salidas del pipeline sobreviven a reinicios de contenedores.

El backend usa un build Docker multi-etapa: las dependencias se compilan en una etapa builder con `uv`, y luego se copian a una imagen runtime con CUDA. La imagen runtime incluye `gcc/g++` porque SAM3 usa PyTorch Triton, que compila kernels CUDA en tiempo de ejecucion (JIT).

---

## Pipeline ML

El pipeline tiene dos enfoques principales que trabajan juntos. Para videos cortos, el pipeline YOLO+SAM3 procesa el video completo directamente. Para videos mas largos (mas de ~80 segundos), un pre-filtro rapido basado en pixeles primero identifica los momentos donde realmente esta pasando algo, y luego el pipeline YOLO+SAM3 procesa solo esos segmentos.

### Fase 1: Pre-filtro SimpleDetector (CPU, rapido)

Para videos largos (por ejemplo, una grabacion de camara de seguridad de 2 horas), ejecutar el pipeline completo de GPU en cada cuadro tomaria una cantidad de tiempo impracticable. El SimpleDetector resuelve esto escaneando rapidamente todo el video en CPU para encontrar los momentos en que los grifos realmente estan siendo usados.

**Como funciona:**

El detector monitorea dos pequenas regiones de interes (ROIs), una alrededor de cada manija de grifo. Para cada cuadro, calcula diferencias de pixeles contra un fondo de referencia. Cuando los pixeles en la region de una manija cambian significativamente, significa que alguien esta tirando de la manija, un vaso se esta moviendo debajo, o el liquido esta fluyendo. Esto produce una senal de actividad simple por grifo: un numero entre 0 y 1 que indica cuanto movimiento hay en esa region.

**Multiprocesamiento para velocidad:** Para videos largos, el detector divide el video en fragmentos y los procesa en paralelo usando multiprocessing de Python. Esto le permite escanear un video de 2 horas en cuestion de segundos en CPU.

**Compromiso de precision:** El SimpleDetector no es lo suficientemente preciso para contar servidos por si solo. Cambios en el fondo, variaciones de iluminacion o personas pasando pueden disparar senales de actividad falsas. Su proposito es puramente como pre-filtro: identifica *cuando* esta pasando algo interesante para que el pipeline YOLO (preciso pero mas lento) solo necesite procesar esos segmentos.

**Ejemplo:** Un video de 2 horas podria generar 8 ventanas de actividad que totalizan 12 minutos de uso real de los grifos. En lugar de ejecutar YOLO sobre 216,000 cuadros, el pipeline procesa solo ~21,600 cuadros -- una reduccion de 10x en tiempo de procesamiento GPU.

Las ventanas de actividad se extraen como clips de video cortos, y cada clip se procesa independientemente por el pipeline YOLO+SAM3. Las marcas de tiempo se mapean de vuelta al video original para que todos los eventos referencien el tiempo correcto.

### Entendiendo la Configuracion ROI

Antes de que pueda ocurrir cualquier procesamiento, el sistema necesita saber *donde* en el cuadro del video estan ubicados los grifos. Cada angulo de camara es diferente, por lo que esto debe configurarse por restaurante y camara.

La configuracion ROI define dos tipos de regiones:

1. **Region de recorte YOLO** (`tap_roi`): Un area rectangular del cuadro que contiene ambos grifos, los vasos y las personas siendo servidas. El modelo YOLO procesa solo esta region recortada, lo que reduce el ruido del resto de la escena (otros clientes, televisores, decoraciones) y acelera la inferencia.

2. **Bounding boxes de manijas SAM3** (`sam3_tap_bboxes`): Dos pequenos bounding boxes, uno para cada manija de grifo, dentro de la region recortada. Estos inicializan el modelo de segmentacion SAM3 para que sepa que objetos rastrear.

Para el pre-filtro SimpleDetector, hay dos ROIs adicionales (`roi_1`, `roi_2`) que definen pequenas regiones alrededor de cada manija para la diferencia de pixeles. Estos son mas simples y solo necesitan cubrir la manija y el area donde fluye el liquido.

Las configuraciones ROI se almacenan como archivos JSON en `data/roi_configs/` y pueden crearse a traves del asistente visual del frontend o las herramientas interactivas de CLI.

### Fase 2: Tracking YOLO (GPU-intensivo)

Esta es la etapa central de deteccion. Usa **YOLO-World** (`yolov8x-worldv2.pt`), un modelo de deteccion de objetos zero-shot con vocabulario abierto. La ventaja clave de YOLO-World es que podemos especificar exactamente que clases de objetos detectar -- en nuestro caso, **"person"** y **"cup"** -- sin necesidad de entrenar un modelo personalizado. Un modelo YOLO estandar entrenado en COCO detectaria docenas de clases irrelevantes (botellas, sillas, televisores, etc.) que agregan ruido y ralentizan el procesamiento. Al enfocarnos en solo dos clases, obtenemos detecciones mas limpias.

**Que detecta:** El modelo busca vasos y personas en la region recortada de los grifos. La senal principal de un evento de servido es la presencia simultanea de una persona y un vaso cerca del grifo -- alguien esta sosteniendo un vaso bajo el grifo para llenarlo.

**Tracking con BoT-SORT:** Las detecciones crudas son cuadro por cuadro, pero necesitamos seguir vasos individuales a traves de multiples cuadros para entender los eventos de servido. BoT-SORT asigna IDs persistentes a los objetos detectados, creando *tracks* que siguen cada vaso desde que aparece hasta que sale del cuadro.

**Salida:** Un archivo CSV (`raw_detections.csv`) con bounding boxes por cuadro, etiquetas de clase, scores de confianza e IDs de track para cada objeto detectado.

**Opciones de optimizacion:** El parametro `sample_every` controla cuantos cuadros se analizan. Configurar `sample_every: 1` procesa cada cuadro (mas preciso). Configurar `sample_every: 3` procesa cada tercer cuadro, reduciendo el tiempo de GPU aproximadamente 3x con minima perdida de precision para la mayoria de videos. Un parametro `record_range` tambien puede limitar el procesamiento a una ventana de tiempo especifica dentro del video.

El tracking YOLO es la etapa mas intensiva en GPU del pipeline. El tiempo de procesamiento depende fuertemente de la resolucion del video, su duracion y la cantidad de objetos en la escena.

### Fase 3: Re-enlace de Tracks y Clasificacion de Servidos

Despues del tracking YOLO, los tracks crudos frecuentemente tienen problemas. El tracker puede perder un objeto por algunos cuadros y luego detectarlo nuevamente con un nuevo ID, creando dos tracks separados para lo que en realidad fue un movimiento continuo de un vaso. O un vaso sentado en la barra esperando ser recogido podria ser rastreado por mucho tiempo sin ser realmente parte de un servido.

La etapa de re-enlace aborda ambos problemas:

**Fusion de tracks fragmentados:** Cuando dos tracks tienen posiciones espaciales similares y no se superponen mucho en el tiempo, probablemente son el mismo objeto. El algoritmo compara tracks basandose en las posiciones de sus bounding boxes y brechas temporales. Si un track termina y otro comienza cerca (dentro de umbrales configurables de pixeles y cuadros), se fusionan en un solo track con un ID. Esto previene el doble conteo de un solo servido que se dividio en multiples fragmentos de track.

**Clasificacion de servidos:** No cada vaso rastreado es parte de un evento de servido. La etapa de re-enlace filtra tracks usando tres criterios:
- **Duracion:** El track debe abarcar suficientes cuadros (`min_pour_frames: 30`) para representar un servido real, no solo una deteccion momentanea.
- **Movimiento:** El vaso debe mostrar suficiente movimiento espacial (`movement_threshold`) -- un vaso siendo llevado hacia y desde el grifo se mueve a traves del cuadro.
- **No estacionario:** Los vasos que permanecen en una posicion durante la mayor parte de su vida rastreada (por ejemplo, un vaso dejado en la barra) se filtran. Si un vaso permanece dentro de un pequeno radio de pixeles de su posicion mediana durante mas del 80% de su tiempo de vida, se clasifica como estacionario y se excluye.

**Escalado consciente de resolucion:** Estos umbrales basados en pixeles fueron calibrados en recortes de ~800px de ancho de video 4K. Cuando se procesa video de menor resolucion (por ejemplo, 360p), la region de recorte es mucho mas pequena en pixeles. La etapa de re-enlace detecta automaticamente el ancho del recorte y escala todos los umbrales proporcionalmente, para que la misma configuracion funcione en diferentes resoluciones de video.

**Margen de mejora:** Los parametros de re-enlace (umbral de superposicion, brecha de interpolacion, umbrales de movimiento) ofrecen un margen significativo de ajuste fino por despliegue. Diferentes angulos de camara, distancias y calidades de video pueden beneficiarse de parametros ajustados. Esta es un area donde el trabajo futuro podria mejorar la precision.

### Fase 4: Tracking de Manijas con SAM3 (GPU, lento)

En este punto, sabemos *cuando* ocurrieron los eventos de servido y tenemos rastreados los vasos involucrados, pero aun no sabemos *de que grifo* vino cada servido. YOLO-World no detecta bien las manijas de grifo -- son objetos metalicos pequenos que no aparecen en vocabularios de deteccion estandar.

Para resolver esto, usamos **SAM3 (Segment Anything Model 3)**, un modelo de segmentacion por instancias que puede segmentar y rastrear objetos arbitrarios a traves de cuadros de video. SAM3 se inicializa con bounding boxes alrededor de cada manija de grifo (de la configuracion ROI), y luego propaga esas segmentaciones a traves del video usando un mecanismo de banco de memoria.

**Que produce:** Para cada cuadro durante un evento de servido, SAM3 genera las coordenadas del centroide (punto central) de la mascara de segmentacion de cada manija. Cuando una manija esta siendo tirada, su centroide se mueve -- particularmente en la direccion vertical (Y).

**Logica de asignacion de grifo:** Para el rango de cuadros de cada evento de servido, el sistema calcula la desviacion estandar de la coordenada Y del centroide de cada manija. El grifo con mas movimiento vertical durante ese servido es el que esta siendo usado. Si ningun grifo muestra movimiento significativo, el servido se marca como `UNKNOWN`.

**Eficiencia:** SAM3 solo procesa los rangos de cuadros donde se detectaron eventos de servido (de la Fase 3), no el video completo. Tambien usa salto de cuadros (`frame_skip: 5`) e inferencia en media precision (`half: true`) para reducir el tiempo de GPU.

**Compromiso:** SAM3 es la etapa mas lenta del pipeline. La segmentacion por instancias con propagacion de memoria es computacionalmente costosa. Sin embargo, proporciona una asignacion de grifo confiable que seria muy dificil de lograr solo con deteccion por bounding boxes.

---

## Backend

El backend FastAPI gestiona las subidas de video, lanza el procesamiento ML y sirve los resultados.

### Tablas de Base de Datos

**Tabla Videos:** Almacena metadatos de cada video subido -- nombre de archivo, fecha de subida, estado de procesamiento (`pending`, `processing`, `completed`, `error`), duracion e informacion de tiempos. Cada video tambien esta asociado a un `restaurant_name` y `camera_id`.

**Tabla TapEvents:** Almacena eventos individuales de servido detectados por el pipeline. Cada evento registra que grifo (`A` o `B`), los numeros de cuadro inicio/fin, marcas de tiempo inicio/fin (en segundos), un score de confianza y un conteo (numero de cervezas en ese evento, tipicamente 1). Los eventos estan vinculados a su video fuente mediante una clave foranea con eliminacion en cascada.

Los conteos de cervezas se calculan como `SUM(count)` por grifo, no `COUNT(*)` de eventos.

### Restaurante e ID de Camara

Cada video se etiqueta con un nombre de restaurante e ID de camara. Esto sirve para dos propositos:

1. **Busqueda de configuracion ROI:** Diferentes camaras tienen diferentes angulos, resoluciones y posiciones. La configuracion ROI (donde estan los grifos en el cuadro) se almacena por combinacion restaurante+camara. Cuando inicia el procesamiento, el sistema carga la configuracion ROI correspondiente automaticamente.

2. **Soporte multi-ubicacion:** Una cadena de bares podria tener multiples locaciones, cada una con una o mas camaras. El etiquetado restaurante+camara mantiene los conteos organizados y asegura que cada video se procese con la calibracion espacial correcta.

### Endpoints de la API

| Metodo | Endpoint | Proposito |
|--------|----------|-----------|
| POST | `/api/videos/upload` | Subir video mp4/mov |
| POST | `/api/videos/{id}/process` | Iniciar procesamiento ML |
| GET | `/api/videos/{id}/status` | Estado + conteos + eventos |
| GET | `/api/videos/` | Listar todos los videos |
| DELETE | `/api/videos/{id}` | Eliminar video + eventos |
| GET | `/api/counts/` | Consultar conteos (filtros: video_id, fecha, grifo) |
| GET | `/api/counts/summary` | Totales agregados |

---

## Frontend

El frontend Streamlit proporciona dos vistas principales:

### Pestana Subir y Procesar

- Seleccionar un restaurante y camara (o crear nuevos).
- Subir un archivo de video. Si ya existe una configuracion ROI para ese restaurante+camara, el procesamiento inicia automaticamente.
- Si no existe configuracion ROI, un asistente visual de 3 pasos te guia para seleccionar la region de recorte, la manija del Grifo A y la manija del Grifo B en el primer cuadro del video.
- Durante el procesamiento, la interfaz muestra el estado actual. Solo un video se procesa a la vez (restriccion de GPU), asi que subidas adicionales se encolan como "pending" y se lanzan automaticamente cuando la GPU esta disponible.
- Al completarse, la pestana muestra el conteo del Grifo A, conteo del Grifo B, total y una tabla de eventos individuales de servido.

### Pestana Dashboard

- Muestra metricas resumen globales: total de servidos por grifo en todos los videos, gran total y numero de videos procesados.
- Lista todos los videos con detalles expandibles, metricas por video y botones de eliminar.

### Salidas Intermedias para Depuracion

Todos los resultados intermedios del pipeline ML se guardan en `data/results/web_{video_id}/` para cada video procesado. Esto incluye:

- **Video anotado de tracking YOLO** (`yolo_raw_tracking.mp4`): Muestra bounding boxes e IDs de track superpuestos en el video recortado, util para verificar que las detecciones y el tracking funcionan correctamente.
- **Detecciones re-enlazadas y graficos**: CSVs y visualizaciones mostrando como se fusionaron los tracks y cuales se clasificaron como servidos vs. filtrados.
- **Trayectorias de centroides SAM3** (`sam3_centroids.csv`): Posiciones de manija por cuadro, utiles para verificar la logica de asignacion de grifo.
- **Archivos JSON de eventos de servido**: Tanto pre-asignacion (`pour_events.json`) como post-asignacion (`pour_events_assigned.json`), mostrando la progresion completa del pipeline.
- **Salidas del SimpleDetector** (para videos largos): Senales de actividad, mapas de calor y clips extraidos.

Estas salidas son valiosas para depurar falsos positivos/negativos, ajustar parametros del pipeline y entender por que el sistema tomo decisiones de conteo especificas.

---

## Testing

```bash
uv run pytest tests/ -v    # 59 tests
uv run ruff check .        # lint
uv run ruff format .       # formatear
```

Los tests usan una base de datos SQLite en memoria y mockean las dependencias ML, por lo que se ejecutan sin GPU.

## Estructura del Proyecto

```
backend/
  main.py              # Punto de entrada FastAPI
  config.py            # Configuraciones, rutas, constantes
  database/            # Modelos SQLAlchemy, schemas, conexion
  routers/             # Endpoints de videos + conteos
  services/            # Procesador en segundo plano (orquesta el pipeline ML)
  ml/
    common.py          # Utilidades compartidas (ROI, recorte, selectores interactivos)
    approach_yolo/     # Pipeline YOLO + SAM3 (4 etapas)
    approach_simple/   # Diferencia de pixeles CPU (pre-filtro + CLI)
frontend/
  app.py               # Interfaz Streamlit (subida, asistente ROI, dashboard)
  utils/api_client.py  # Cliente HTTP del backend
config/                # Configs YAML del pipeline + config del tracker BoT-SORT
data/                  # Videos, modelos, BD, configs ROI, resultados
tests/                 # Suite pytest (59 tests)
scripts/               # Herramientas CLI para ejecutar etapas del pipeline independientemente
```
