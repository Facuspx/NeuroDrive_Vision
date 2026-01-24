NeuroDrive_Vision: Módulo de detección de somnolencia 

Descripción general

NeuroDrive_Vision es el módulo encargado de la detección de somnolencia del conductor dentro del proyecto NeuroDrive.
Su función es analizar señales visuales del rostro (ojos, boca y gestos asociados a fatiga), extraer eventos discretos y prepararlos para su evaluación por una máquina de estados.

Este módulo no toma decisiones finales ni gestiona actuadores directamente. Su responsabilidad es generar información confiable, temporalmente validada y estructurada para un sistema de decisión de nivel superior.

Objetivo del módulo

Detectar indicadores visuales de somnolencia mediante visión artificial.

Convertir mediciones continuas en eventos discretos temporizados.

Evitar falsas detecciones mediante ventanas temporales y contadores.

Entregar señales limpias y robustas a la máquina de estados TD3.

Enfoque de diseño

El diseño sigue principios de sistemas digitales robustos:

Separación estricta entre captura, medición, temporalidad y decisión.

Evaluación basada en persistencia de eventos, no en muestras aisladas.

Arquitectura modular y escalable.

Preparación explícita para integración con una máquina de estados finita.

Operación local (offline), orientada a sistemas embebidos.

Estructura del proyecto
NeuroDrive_Vision/
│
├─ entorno/.venv
├─ neurodrive_vision/
│   ├─ __init__.py
│   ├─ captura_video.py
│   ├─ detector_rostro_mediapipe.py
│   ├─ medidas_rostro.py
│   ├─ contador_eventos.py
│   ├─ detector_frote_ojos.py
│   ├─ reporte_simplificado.py
│   └─ integracion_maquina_estados.py
│
├─README.md
├─requirements.txt
└─ main.py

Descripción de los módulos

captura_video.py
Adquisición de frames de video desde la cámara. Aísla la fuente de video del resto del sistema.

detector_rostro_mediapipe.py
Detección del rostro y landmarks faciales utilizando MediaPipe.

medidas_rostro.py
Cálculo de métricas geométricas del rostro (EAR, MAR y variables auxiliares).

contador_eventos.py
Implementación de lógica temporal y ventanas deslizantes para validar eventos de somnolencia.

detector_frote_ojos.py
Detección del gesto de frotarse los ojos como indicador adicional de fatiga.

reporte_simplificado.py
Generación de reportes técnicos simplificados y logging del estado del sistema.

integracion_maquina_estados.py
Adaptación de eventos y contadores para su ingreso a la máquina de estados TD3.

main.py
Script principal para pruebas locales e integración del módulo completo.