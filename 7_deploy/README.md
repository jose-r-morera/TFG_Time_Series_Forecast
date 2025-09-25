# Sistema de Predicción Meteorológica de Canarias

Este proyecto full-stack permite la visualización y análisis de datos meteorológicos obtenidos de estaciones distribuidas en las Islas Canarias. Permite además lanzar predicciones automáticas de corto plazo mediante un modelo secuencial LSTM previamente entrenado, que se ejecuta de forma asíncrona para no bloquear la experiencia del usuario y permitir un cierto nivel de paralelismo.

El sistema está dividido en dos partes: un frontend interactivo desarrollado con Next.js y React, y un backend en FastAPI que gestiona tareas con Celery y Redis.

---

## 📂 Estructura General del Proyecto

### Frontend (Next.js + React)

El frontend está diseñado como una SPA (Single Page Application) que permite seleccionar estaciones meteorológicas, consultar gráficas de sensores y lanzar peticiones de predicción.

**Características principales:**

* Selector de estaciones agrupadas por ubicación.
* Visualización de sensores disponibles por estación.
* Indicadores visuales del estado de la predicción (cargando, fallida, completada).
* Componente visual de progreso y gestión de los estados erroneos.

**Componentes principales del aplicativo:**

| Archivo                         | Descripción                                                                                                  |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `weather-station-dashboard.tsx` | Componente raíz: gestión de estado global, selección, carga de datos, renderizado de gráficas y predicciones |
| `station-selector.tsx`          | Selector de estaciones meteorológicas ordenadas por ubicación geográfica                                     |
| `sensor-chart.tsx`              | Renderizado de gráficas con datos históricos y resultados de predicción                                      |
| `prediction-status.tsx`         | Indicador de estado para tareas de predicción activas o completadas                                          |
| `loading-indicator.tsx`         | Componente de carga y visualización del progreso de carga                                                    |
| `no-data-message.tsx`           | Mensaje mostrado cuando no hay observaciones disponibles                                                     |
| `stylish-header.tsx`            | Cabecera con navegación y selector de estación                                                               |
| `station-info.tsx`              | Información detallada de la estación seleccionada                                                            |

---

### Backend (FastAPI + Celery)

El backend expone una API REST encargada de gestionar las solicitudes de predicción, transformar los datos recibidos, generar las entradas necesarias para el modelo y coordinar su ejecución.

**Endpoints disponibles:**

* `POST /predict`: Recibe datos estructurados de una estación con observaciones recientes, genera el tensor de entrada y encola una tarea en Celery para su predicción.
* `GET /predict/{job_id}`: Permite consultar el estado en el que se encuentra la predicción y, en el caso de que este disponible el resultado correspondiente al `job_id` proporcionado.

**Procesamiento interno:**

* Extracción de características temporales (senos y cosenos del ciclo diario y anual).
* Conversión de los datos recibidos a tensores NumPy de dimensión `[1, T, F]`, donde `T=??` y `F=7` (4 features temporales + 3 sensores).
* Predicción mediante un modelo LSTM entrenado, cuyos pesos se cargan en la inicialización del worker.

**Worker Celery:**

* El worker se encarga de ejecutar las predicciones de forma independiente al servidor web.
* Carga el modelo desde `weights/????` en el arranque.
* Usa Redis como broker de tareas y almacenamiento de resultados.

---

## 🚀 Flujo de Predicción

1. El usuario selecciona una estación y lanza la predicción para un sensor.
2. El frontend recoge los datos de observación de los últimos 48 intervalos horarios.
3. El paquete de datos se estructura y se envía al backend vía `POST /predict`.
4. FastAPI valida, transforma y encola la tarea en Celery.
5. Celery ejecuta el modelo LSTM y guarda el resultado.
6. El frontend consulta periódicamente `GET /predict/{job_id}` hasta obtener el resultado.
7. Las predicciones se visualizan sobre las gráficas originales del sensor.

---

## 🚚 Docker Compose

```yaml
services:
  frontend:
    build: ./portal_inferencia
    ports: ["3000:3000"]

  backend:
    build: ./backend_api
    ports: ["8000:8000"]
    environment:
      - REDIS_HOST=redis
      - REDIS_PORT=6379

  celery:
    build: ./backend_api
    command: celery -A celery_worker worker --loglevel=info
    depends_on: [redis]
    environment:
      - REDIS_HOST=redis
      - REDIS_PORT=6379

  redis:
    image: redis:latest
```

---

## 🔧 Librería `/lib`

Los componentes listados en el front hacen uso de los siguientes modulos complementarios desarrollados. 

| Archivo                 | Propósito                                                                                |
| ----------------------- | ---------------------------------------------------------------------------------------- |
| `api.ts`                | Abstracción de llamadas a la API de Grafcan con manejo de errores                        |
| `chart-setup.ts`        | Estilos globales, configuraciones y ejes para Chart.js                                   |
| `chart-utils.ts`        | Lógica para tooltips, leyendas y formato dinámico de etiquetas                           |
| `data-processing.ts`    | Agregación de observaciones a nivel horario, cálculo de medias y estructuras de entrada  |
| `prediction-service.ts` | Comunicación directa con el backend: `POST /predict` y polling a `GET /predict/{job_id}` |
| `types.ts`              | Definiciones de tipos TypeScript para entidades: estaciones, sensores, observaciones     |
| `utils.ts`              | Funciones utilitarias comunes como formateo de fechas y etiquetas                        |

---

## 📊 API de Datos en Tiempo Real

* API SensorThings: `https://sensores.grafcan.es/api/v1.0`
* Estándar OGC para observaciones y sensores IoT.
* Acceso autorizado mediante API key: `Authorization: Api-Key <API_KEY>`

---

## 👁️ Visualización de Entrada al Modelo

Durante la fase de predicción, el backend genera las siguientes imágenes para depuración, donde cada imagen representa la evolución temporal de uno de los canales de entrada al modelo:

```
debug_inputs/
├── <job_id>_channel_1.png  # sin_day
├── <job_id>_channel_2.png  # cos_day
├── <job_id>_channel_3.png  # sin_week
├── <job_id>_channel_4.png  # cos_week
├── <job_id>_channel_5.png  # temperatura
├── <job_id>_channel_6.png  # humedad
├── <job_id>_channel_7.png  # presión
```

Estas gráficas permiten verificar que la entrada al modelo fue construida correctamente.


---


## Pruebas de carga con Locust
El sistema puede someterse a pruebas de carga usando Locust, una herramienta open-source para simular múltiples usuarios concurrentes que ejecutan tareas programadas.

* Se pueden lanzar múltiples tareas que simulan la creación de predicciones (POST /predict) y luego realizar consultas periódicas (GET /predict/{job_id}) para monitorizar su estado.

* Permite medir el rendimiento del sistema, identificar cuellos de botella y verificar la estabilidad bajo carga.


---


## Escalabilidad y futuro
Aunque el despliegue actual es sencillo y con capacidad limitada de paralelización (un único worker), el diseño es completamente escalable:

- Celery permite múltiples workers, por lo tanto basta con lanzar más réplicas del contenedor del worker para aumentar la capacidad de procesamiento.

- Con un orquestador como Docker Swarm o Kubernetes, se puede escalar horizontalmente de forma automática.

- Se puede implementar un mecanismo sencillo de arbitraje (como balanceo por colas o prioridad por tipo de tarea) si el volumen de predicciones o su latencia aumenta.


---


## 🚧 Requisitos Técnicos

* Docker y Docker Compose (para entorno de ejecución completo)
* Node.js >= 18.x (para desarrollo y testing del frontend)
* Python 3.11 con librerías:

  * `fastapi`, `pydantic`, `uvicorn`
  * `celery`, `redis`, `numpy`, `matplotlib`
  * `tensorflow` (versión compatible con el modelo entrenado)


---

## 📍 Autor

Desarrollado por JR.

El proyecto implementa una arquitectura desacoplada, con procesamiento asincrónico, aprendizaje automático y visualización meteorológica con APIs estándar abiertas.

