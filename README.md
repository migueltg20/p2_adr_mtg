# Filtros de Kalman para Estimación de Pose y Velocidad

Este repositorio contiene dos implementaciones diferentes de filtros de Kalman para la estimación de estados de un robot:

## 1. Filtro de Kalman para Estimación de Pose (`kf_estimation.py`)

Este filtro implementa un estimador básico que rastrea la posición y orientación del robot en 2D.

### Características:
- **Vector de estado**: 3 dimensiones `[x, y, θ]` (posición x, posición y, orientación)
- **Entradas de control**: Velocidad lineal (v) y velocidad angular (ω)
- **Modelo de movimiento**: Modelo de velocidad simple que actualiza la posición basándose en la orientación actual
- **Mediciones**: Posición y orientación ruidosas del odómetro

### Funcionamiento:
1. **Predicción**: Utiliza el modelo de movimiento para predecir la nueva posición basada en la velocidad
2. **Actualización**: Corrige la predicción utilizando las mediciones del odómetro
3. **Publicación**: Publica la estimación como un mensaje `PoseWithCovarianceStamped`

## 2. Filtro de Kalman para Estimación de Pose y Velocidad (`kf_estimation_vel.py`)

Este filtro implementa un estimador avanzado que rastrea tanto la posición como la velocidad del robot.

### Características:
- **Vector de estado**: 6 dimensiones `[x, y, θ, vx, vy, ω]` (posición, orientación, velocidades lineales en x e y, velocidad angular)
- **Entradas de control**: Velocidad lineal (v) y velocidad angular (ω)
- **Modelo de movimiento**: Modelo de velocidad constante que actualiza posición y velocidad
- **Mediciones**: Posición, orientación y componentes de velocidad con ruido

### Funcionamiento:
1. **Predicción**: Utiliza el modelo de movimiento de velocidad constante para predecir el nuevo estado
2. **Actualización**: Corrige la predicción utilizando mediciones completas (posición y velocidad)
3. **Publicación**: Publica la estimación como un mensaje `PoseWithCovarianceStamped`

## Diferencias principales

- **Complejidad del estado**: El segundo filtro (`kf_estimation_vel`) mantiene más información al rastrear explícitamente las velocidades
- **Precisión**: El filtro de velocidad suele proporcionar estimaciones más precisas al modelar directamente la dinámica de velocidad
- **Robustez**: El filtro extendido es más robusto ante cambios rápidos en la velocidad

## Utilización

Para ejecutar cualquiera de los nodos:

```bash
ros2 run p2_adr_mtg kf_estimation           # Para el filtro básico
ros2 run p2_adr_mtg kf_estimation_vel       # Para el filtro con velocidad
```

La visualización de las estimaciones está disponible usando RViz o mediante visualizadores específicos incluidos en el paquete.

## Estructura de clases y dependencias

Ambos filtros dependen de:
- Modelos de movimiento (`motion_models.py`) 
- Modelos de observación (`observation_models.py`)
- Implementación principal del filtro (`kalman_filter.py`)
- Utilidades para sensores (`sensor_utils.py`)
- Visualizadores (`simple_visualizer.py` o `visualization.py`)

La arquitectura está diseñada de forma modular para facilitar la extensión y adaptación a diferentes configuraciones de robots.
