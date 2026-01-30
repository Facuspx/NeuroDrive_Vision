import logging
import cv2

from neurodrive_vision.captura_video import CapturadorVideo, ErrorCapturaVideo
from neurodrive_vision.detector_rostro_mediapipe import (DetectorRostroMediaPipe, ErrorInicializacionDetector)


def configurar_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
    )


def main():
    configurar_logging()
    logger = logging.getLogger("NeuroDriveMain")

    # Inicializar detector de rostro (MediaPipe)
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
        logger.error(f"No se pudo inicializar el DetectorRostroMediaPipe: {e}")
        return

    try:
        with CapturadorVideo(
            indice_camara=0,
            ruta_video="video_example.mp4",
            resolucion=(640, 480),
            usar_csi=False,
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

                # --- FRAME ORIGINAL (sin dibujo) ---
                frame_original = frame.copy()

                # --- PROCESAR ROSTRO ---
                datos_rostro = detector_rostro.procesar_frame(frame)

                # --- GENERAR MÁSCARA OSCURA CON PUNTOS ---
                mascara = detector_rostro.dibujar_malla(
                    frame_bgr=frame,
                    datos_rostro=datos_rostro,
                    dibujar_contornos=False,
                    dibujar_puntos=True,
                )

                # --- TEXTO SOBRE LA MÁSCARA ---
                metricas = detector_rostro.obtener_metricas()
                reporte = metricas.obtener_reporte()

                texto_fps = f"FPS detector: {reporte['fps_promedio']:.1f}"
                texto_tasa = f"Deteccion: {reporte['tasa_deteccion_pct']:.1f}%"
                texto_rostro = "Rostro: SI" if datos_rostro.rostro_presente else "Rostro: NO"

                cv2.putText(
                    mascara,
                    texto_fps,
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    mascara,
                    texto_tasa,
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    mascara,
                    texto_rostro,
                    (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                # --- MOSTRAR VENTANAS ---
                cv2.imshow("NeuroDrive - Frame Original", frame_original)
                cv2.imshow("NeuroDrive - Mascara Rostro", mascara)

                # Tecla 'q' para salir
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except ErrorCapturaVideo as e:
        logger.error(f"Error en la captura de video: {e}")

    finally:
        detector_rostro.liberar()
        cv2.destroyAllWindows()

        metricas_finales = detector_rostro.obtener_metricas()
        logger.info("=== Métricas finales del detector de rostro ===")
        for clave, valor in metricas_finales.obtener_reporte().items():
            logger.info(f"{clave}: {valor}")


if __name__ == "__main__":
    main()
