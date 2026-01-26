import cv2
from neurodrive_vision.captura_video import CapturadorVideo, ErrorCapturaVideo


def main():
    try:
        capturador = CapturadorVideo(indice_camara=0, resolucion=(640, 480))
        capturador.iniciar()
        print("Fuente de video iniciada correctamente.")
        print("FPS reportados:", capturador.obtener_fps())
        print("Resolución actual:", capturador.obtener_resolucion())
    except ErrorCapturaVideo as e:
        print(f"[ERROR] {e}")
        return

    while True:
        ok, frame = capturador.leer_frame()
        if not ok:
            print("[ADVERTENCIA] No se pudo leer frame. ¿Se terminó el video o se desconectó la cámara?")
            break

        cv2.imshow("Prueba captura NeuroDrive", frame)

        # Tecla 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capturador.liberar()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
