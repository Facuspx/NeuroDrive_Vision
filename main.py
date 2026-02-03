import logging
import cv2
import numpy as np
from neurodrive_vision.captura_video import CapturadorVideo, ErrorCapturaVideo
from neurodrive_vision.detector_rostro_mediapipe import (DetectorRostroMediaPipe,ErrorInicializacionDetector,)
from neurodrive_vision.medidas_rostro import CalculadorMedidasRostro
from neurodrive_vision.contador_eventos import ContadorEventosSomnolencia
from neurodrive_vision.detector_frote_ojos import DetectorFroteOjosMediaPipe
from neurodrive_vision.integracion_maquina_estados import IntegradorMaquinaEstados



def configurar_logging():
    logging.basicConfig(
        level=logging.INFO,
        format=" %(levelname)s - %(name)s - %(message)s",
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
    detector_frote = DetectorFroteOjosMediaPipe()
    integrador_maquina = IntegradorMaquinaEstados()

    # ----- Inicializar captura de video -----
    try:
        with CapturadorVideo(
            indice_camara= 1,
            ruta_video="video_example.mp4",#"video_example.mp4"
            resolucion=(1280, 720), #(640, 480) (1280, 720)
            usar_csi=False,   # en PC: False; en RPi con cámara CSI: True si configuraste el pipeline
            fps_deseado=30,
        ) as capturador:

            logger.info("Captura de video iniciada correctamente.")
            # logger.info(f"Resolución real: {capturador.obtener_resolucion()}")
            # logger.info(f"FPS reportados: {capturador.obtener_fps()}")

            while True:
                ok, frame = capturador.leer_frame()
                if not ok:
                    logger.warning("No se pudo leer frame. Saliendo del loop.")
                    break

                frame_original = frame.copy()
                contador_eventos.set_frecuencia_cardiaca_simulada(55.0)  # “modo sueño” (85.0)  # más alerta / activo
                # ----- Detección de rostro + puntos -----
                datos_rostro = detector_rostro.procesar_frame(frame)

                resultado_frote = detector_frote.procesar_frame(
                    frame_bgr=frame,
                    datos_rostro=datos_rostro,
                    timestamp=datos_rostro.timestamp,
                )
                
                # ----- Crear máscara negra -----
                mascara = np.zeros_like(frame)

                # Dibujar puntos del rostro sobre la máscara negra
                mascara = detector_rostro.dibujar_malla(
                    frame_bgr = mascara,             
                    datos_rostro = datos_rostro,
                    dibujar_contornos = False,
                    dibujar_puntos = True,
                    color_contorno = (192, 255, 48),
                )

                # Dibujar debug de frote (círculos de ojos y puntas de dedos) en la misma máscara
                mascara = detector_frote.dibujar_debug_sobre_mascara(mascara)


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
                    salida = contador_eventos.actualizar(
                    timestamp=datos_rostro.timestamp,
                    medidas=medidas,
                    resultado_frote=resultado_frote, 
                    )

                    # Estadísticas acumuladas
                    stats = contador_eventos.obtener_estadisticas()

                    salida_maquina = integrador_maquina.actualizar(salida)


                    # ----- Dibujar textos sobre la MÁSCARA -----

                    texto_frote = f"Frote activo: {int(resultado_frote.frote_activo)}"
                    #texto_frote2 = f"Inic/Fin: {int(resultado_frote.frote_iniciado)}/{int(resultado_frote.frote_finalizado)}"
                    texto_frote3 = f"Dedo ojo izq/der: {int(resultado_frote.mano_cerca_ojo_izquierdo)}/{int(resultado_frote.mano_cerca_ojo_derecho)}"

                    cv2.putText(mascara, f"Estado alerta: {salida_maquina.estado_alerta.name}", (10, 610), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1, cv2.LINE_AA,
                    )
                    cv2.putText(mascara, f"Nivel alerta: {salida_maquina.nivel_alerta}", (10, 640), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1, cv2.LINE_AA,
                    )                  


                    cv2.putText(mascara, f"Ventana confiable: {int(not salida.ventana_no_confiable)}",
                        (10, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (192, 255, 48), 1,cv2.LINE_AA,
                    )

                    cv2.putText(mascara, texto_frote, (10, 540), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (192, 255, 48), 2, cv2.LINE_AA,
                    )
                    # cv2.putText( mascara,texto_frote2, (10, 435), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    #     (192, 255, 48), 1, cv2.LINE_AA,
                    # )
                    cv2.putText( mascara, texto_frote3, (10, 570), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (192, 255, 48), 1, cv2.LINE_AA,
                    )

                    # EAR / MAR
                    cv2.putText( mascara, texto_ear, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (192, 255, 48), 2, cv2.LINE_AA,
                    )
                    cv2.putText(mascara, texto_mar, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (192, 255, 48), 2,cv2.LINE_AA,
                    )

                    # Contadores de eventos
                    cv2.putText(mascara, f"Parpadeos: {stats['parpadeos_total']}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (192, 255, 48), 2, cv2.LINE_AA,
                    )
                    cv2.putText(mascara, f"Microsuenos: {stats['microsuenos_total']}", (10, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (192, 255, 48), 2, cv2.LINE_AA,
                    )
                    cv2.putText(mascara, f"Bostezos: {stats['bostezos_total']}", (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (192, 255, 48), 2, cv2.LINE_AA,
                    )
                    cv2.putText(mascara, f"Cabeceos: {stats['cabeceos_total']}", (10, 155),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (192, 255, 48), 2, cv2.LINE_AA,
                    )

                    # Atención
                    cv2.putText(mascara, f"Atencion: {salida.atencion.categoria} ({salida.atencion.nivel:.2f})",
                        (10, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (192, 255, 48), 2, cv2.LINE_AA,
                    )

                    # Mensaje de motivo
                    cv2.putText(mascara, f"{salida.atencion.motivo[:50]}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (192, 255, 48), 1, cv2.LINE_AA,
                    )
                    
                else:
                    # No hay rostro -> solo texto de aviso en la máscara
                    cv2.putText(mascara, "Sin rostro detectado", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255),2, cv2.LINE_AA,
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
