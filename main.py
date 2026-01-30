"""
Integración de:
- captura_video (CapturadorVideo)
- detector_rostro_mediapipe (DetectorRostroMediaPipe)
- medidas_rostro (CalculadorMedidasRostro)
- contador_eventos (ContadorEventosSomnolencia)

Objetivo:
Ver en tiempo real si está contando bien:
- Parpadeos
- Microsueños
- Cabeceos
y mostrar un estimador simple de atención.

Pulsa 'q' para salir.
"""

import logging
import cv2

from neurodrive_vision.captura_video import CapturadorVideo, ErrorCapturaVideo
from neurodrive_vision.detector_rostro_mediapipe import (
    DetectorRostroMediaPipe,
    ErrorInicializacionDetector,
)
from neurodrive_vision.medidas_rostro import CalculadorMedidasRostro
from neurodrive_vision.contador_eventos import ContadorEventosSomnolencia


def configurar_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
    )


def main():
    configurar_logging()
    logger = logging.getLogger("NeuroDriveMain")

    # ----- Inicializar módulos de visión -----
    try:
        detector_rostro = DetectorRostroMediaPipe(
            max_rostros=1,
            confianza_minima_deteccion=0.5,
            confianza_minima_seguimiento=0.5,
            refinar_contornos=True,
            modelo_complejidad=1,
            habilitar_cache=True,
            max_frames_sin_deteccion=5,
        )
    except ErrorInicializacionDetector as e:
        logger.error(f"No se pudo inicializar DetectorRostroMediaPipe: {e}")
        return

    calculador_medidas = CalculadorMedidasRostro()
    contador_eventos = ContadorEventosSomnolencia()

    # ----- Inicializar captura de video -----
    try:
        with CapturadorVideo(
            indice_camara=1,
            ruta_video=None,#"video_example.mp4"
            resolucion=(640, 480),
            usar_csi=False,   # en PC: False; en RPi con cámara CSI: True si configuraste el pipeline
            fps_deseado=30,
        ) as capturador:

            logger.info("Captura de video iniciada correctamente.")
            logger.info(f"Resolución real: {capturador.obtener_resolucion()}")
            logger.info(f"FPS reportados: {capturador.obtener_fps()}")

            while True:
                ok, frame = capturador.leer_frame()
                if not ok:
                    logger.warning("No se pudo leer frame. Saliendo del loop.")
                    break

                frame_original = frame.copy()

                # ----- Detección de rostro + puntos -----
                datos_rostro = detector_rostro.procesar_frame(frame)

                # Generamos la máscara negra con puntos (aunque no haya rostro, devuelve negro)
                mascara = detector_rostro.dibujar_malla(
                    frame_bgr=frame,
                    datos_rostro=datos_rostro,
                    dibujar_contornos=False,
                    dibujar_puntos=True,
                    color_contorno=(0, 255, 255),
                )

                # Valores por defecto para textos
                texto_ear = "EAR: N/A"
                texto_mar = "MAR: N/A"

                # ----- Cálculo de medidas geométricas -----
                if datos_rostro.rostro_presente:
                    medidas = calculador_medidas.calcular_medidas(datos_rostro)

                    if medidas.medidas_ojos.valido and medidas.medidas_ojos.ear_promedio is not None:
                        texto_ear = f"EAR prom: {medidas.medidas_ojos.ear_promedio:.3f}"

                    if medidas.medidas_boca.valido and medidas.medidas_boca.mar is not None:
                        texto_mar = f"MAR: {medidas.medidas_boca.mar:.3f}"

                    # ----- Actualizar contador de eventos -----
                    salida = contador_eventos.actualizar(datos_rostro.timestamp, medidas)

                    # Estadísticas acumuladas
                    stats = contador_eventos.obtener_estadisticas()

                    # ----- Dibujar textos sobre la MÁSCARA -----
                    # EAR / MAR
                    cv2.putText(
                        mascara,
                        texto_ear,
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        mascara,
                        texto_mar,
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    # Contadores de eventos
                    cv2.putText(
                        mascara,
                        f"Parpadeos: {stats['parpadeos_total']}",
                        (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        mascara,
                        f"Microsuenos: {stats['microsuenos_total']}",
                        (10, 105),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        mascara,
                        f"Bostezos: {stats['bostezos_total']}",
                        (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        mascara,
                        f"Cabeceos: {stats['cabeceos_total']}",
                        (10, 155),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    # Atención
                    cv2.putText(
                        mascara,
                        f"Atencion: {salida.atencion.categoria} ({salida.atencion.nivel:.2f})",
                        (10, 185),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    # Mensaje de motivo
                    cv2.putText(
                        mascara,
                        f"{salida.atencion.motivo[:50]}",
                        (10, 210),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                    # ----- Eventos instantáneos (flash grande) -----
                    y_evento = 260
                    if salida.eventos.parpadeo:
                        cv2.putText(
                            mascara,
                            "PARPADEO",
                            (10, y_evento),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        y_evento += 30

                    if salida.eventos.microsueno:
                        cv2.putText(
                            mascara,
                            "MICROSUENO!",
                            (10, y_evento),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        y_evento += 30

                    if salida.eventos.bostezo:
                        cv2.putText(
                            mascara,
                            "BOSTEZO",
                            (10, y_evento),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        y_evento += 30

                    if salida.eventos.cabeceo:
                        cv2.putText(
                            mascara,
                            "CABECEO",
                            (10, y_evento),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        y_evento += 30

                else:
                    # No hay rostro -> solo texto de aviso en la máscara
                    cv2.putText(
                        mascara,
                        "Sin rostro detectado",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

                # ----- Mostrar ventanas -----
                cv2.imshow("NeuroDrive - Frame Original", frame_original)
                cv2.imshow("NeuroDrive - Mascara Eventos", mascara)

                # Tecla 'q' para salir
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except ErrorCapturaVideo as e:
        logger.error(f"Error en la captura de video: {e}")

    finally:
        detector_rostro.liberar()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
