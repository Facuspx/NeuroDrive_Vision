from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List
from .detector_rostro_mediapipe import DatosRostro
import cv2
import numpy as np
import logging
logger = logging.getLogger(__name__)

# Importación condicional de MediaPipe
try:
    import mediapipe as mp
    MEDIAPIPE_MANOS_DISPONIBLE = True
except ImportError:
    MEDIAPIPE_MANOS_DISPONIBLE = False
    mp = None  # type: ignore


# ==============================
#   Estructuras de datos
# ==============================

@dataclass
class ResultadoFroteOjos:
    """Resultado de la detección de frote de ojos para un frame."""
    frote_activo: bool = False
    frote_iniciado: bool = False
    frote_finalizado: bool = False
    duracion_frote_actual: float = 0.0
    mano_cerca_ojo_izquierdo: bool = False   # en realidad: dedos cerca de ojo izq.
    mano_cerca_ojo_derecho: bool = False     # dedos cerca de ojo der.
    confiabilidad: float = 0.0               # 0–1 (heurístico)


# ==============================
#   Clase principal
# ==============================

class DetectorFroteOjosMediaPipe:

    # Índices FaceMesh para ojos (los mismos que usamos para EAR)
    _INDICES_OJO_IZQ = [33, 160, 158, 133, 153, 144]
    _INDICES_OJO_DER = [362, 385, 387, 263, 373, 380]

    # Índices MediaPipe Hands para puntas de dedos
    _INDICES_PUNTAS_DEDOS = [4, 8, 12, 16, 20]

    def __init__(
        self,
        factor_radio_ojo: float = 1.5,
        duracion_min_frote: float = 0.4,
        duracion_max_frote: float = 5.0,
        min_frames_dedo_cerca: int = 3,
        ventana_rostro_valido: float = 1.0,
    ) -> None:

        if not MEDIAPIPE_MANOS_DISPONIBLE:
            raise RuntimeError(
                "MediaPipe Hands no está disponible. "
                "Instala mediapipe para usar DetectorFroteOjosMediaPipe."
            )

        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        # Parámetros
        self._factor_radio_ojo = factor_radio_ojo
        self._dur_min_frote = duracion_min_frote
        self._dur_max_frote = duracion_max_frote
        self._min_frames_dedo_cerca = min_frames_dedo_cerca
        self._ventana_rostro_valido = ventana_rostro_valido

        # Estado interno frote
        self._frote_activo: bool = False
        self._duracion_frote_actual: float = 0.0
        self._frames_dedo_cerca_consecutivos: int = 0

        # Ojos (última posición conocida)
        self._centro_ojo_izq: Optional[Tuple[float, float]] = None
        self._centro_ojo_der: Optional[Tuple[float, float]] = None
        self._radio_ojo_izq: Optional[float] = None
        self._radio_ojo_der: Optional[float] = None
        self._ultimo_timestamp_rostro: Optional[float] = None

        # Debug
        self._ultimas_puntas_dedos_px: List[Tuple[int, int]] = []

        logger.info("DetectorFroteOjosMediaPipe inicializado (puntas de dedos).")

    # ---------- API principal ----------

    def procesar_frame(
        self,
        frame_bgr: np.ndarray,
        datos_rostro: DatosRostro,
        timestamp: float,
        dt: Optional[float] = None,
    ) -> ResultadoFroteOjos:
        
        if dt is None:
            dt = 1.0 / 30.0

        resultado = ResultadoFroteOjos()

        if frame_bgr is None or frame_bgr.size == 0:
            logger.warning("Frame vacío en DetectorFroteOjosMediaPipe.")
            self._reset_estado()
            return resultado

        alto, ancho = frame_bgr.shape[:2]

        # 1) Actualizar posición de ojos (si hay rostro)
        if datos_rostro.rostro_presente and datos_rostro.puntos_pixeles is not None:
            self._actualizar_ojos(datos_rostro.puntos_pixeles, timestamp)

        # 2) Verificar si aún tenemos ojos válidos (aunque este frame no vea el rostro)
        ojos_validos = (
            self._centro_ojo_izq is not None
            and self._centro_ojo_der is not None
            and self._radio_ojo_izq is not None
            and self._radio_ojo_der is not None
            and self._ultimo_timestamp_rostro is not None
            and (timestamp - self._ultimo_timestamp_rostro) <= self._ventana_rostro_valido
        )

        if not ojos_validos:
            # Sin referencia de ojos, no podemos hablar de frote
            self._reset_estado()
            return resultado

        # 3) Detectar puntas de dedos con MediaPipe Hands
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        resultados_manos = self._hands.process(frame_rgb)

        self._ultimas_puntas_dedos_px = []

        mano_cerca_izq = False
        mano_cerca_der = False

        if resultados_manos.multi_hand_landmarks:
            for hand_landmarks in resultados_manos.multi_hand_landmarks:
                for idx in self._INDICES_PUNTAS_DEDOS:
                    lm = hand_landmarks.landmark[idx]
                    x_px = int(lm.x * ancho)
                    y_px = int(lm.y * alto)
                    self._ultimas_puntas_dedos_px.append((x_px, y_px))

                    # Distancias a ojos
                    if self._dedo_en_ojo((x_px, y_px), ojo_izquierdo=True):
                        mano_cerca_izq = True
                    if self._dedo_en_ojo((x_px, y_px), ojo_izquierdo=False):
                        mano_cerca_der = True

        dedo_cerca_algun_ojo = mano_cerca_izq or mano_cerca_der

        # 4) Actualizar estado de frote según dedos cerca + duración
        if dedo_cerca_algun_ojo:
            self._frames_dedo_cerca_consecutivos += 1
            if self._frote_activo:
                # Frote ya confirmado, seguimos sumando tiempo
                self._duracion_frote_actual += dt
            else:
                # Posible inicio de frote
                if self._frames_dedo_cerca_consecutivos >= self._min_frames_dedo_cerca:
                    # Confirmamos inicio
                    self._frote_activo = True
                    self._duracion_frote_actual = dt
                    resultado.frote_iniciado = True
        else:
            # No hay dedos cerca del ojo
            if self._frote_activo:
                # Veníamos de un frote, evaluamos duración
                if self._duracion_frote_actual >= self._dur_min_frote:
                    resultado.frote_finalizado = True
                # Cerramos evento
                self._frote_activo = False
                self._duracion_frote_actual = 0.0
            self._frames_dedo_cerca_consecutivos = 0

        # 5) Evitar frote infinito
        if self._frote_activo and self._duracion_frote_actual >= self._dur_max_frote:
            resultado.frote_finalizado = True
            self._frote_activo = False
            self._duracion_frote_actual = 0.0
            self._frames_dedo_cerca_consecutivos = 0

        # 6) Completar salida
        resultado.frote_activo = self._frote_activo
        resultado.duracion_frote_actual = self._duracion_frote_actual
        resultado.mano_cerca_ojo_izquierdo = mano_cerca_izq
        resultado.mano_cerca_ojo_derecho = mano_cerca_der

        # Confiabilidad: dedo en ojo izq/der -> 0.7, frote activo -> 0.9
        if dedo_cerca_algun_ojo and self._frote_activo:
            resultado.confiabilidad = 0.9
        elif dedo_cerca_algun_ojo:
            resultado.confiabilidad = 0.7
        else:
            resultado.confiabilidad = 0.0

        return resultado

    # ---------- Lógica de ojos ----------

    def _actualizar_ojos(
        self,
        puntos_pixeles: List[Tuple[int, int]],
        timestamp: float,
    ) -> None:
        """
        Calcula centro y radio aproximado de cada ojo a partir de puntos FaceMesh.
        """
        def centro_y_radio(indices: List[int]) -> Tuple[Tuple[float, float], float]:
            xs = []
            ys = []
            for idx in indices:
                if 0 <= idx < len(puntos_pixeles):
                    x, y = puntos_pixeles[idx]
                    xs.append(float(x))
                    ys.append(float(y))
            if not xs or not ys:
                raise ValueError("No se pudieron obtener puntos de ojo para frote.")

            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)

            # Radio básico: mitad de la distancia horizontal aprox
            xs_sorted = sorted(xs)
            ancho_ojo = xs_sorted[-1] - xs_sorted[0]
            radio = (ancho_ojo / 2.0) * self._factor_radio_ojo

            return (cx, cy), radio

        try:
            centro_izq, radio_izq = centro_y_radio(self._INDICES_OJO_IZQ)
            centro_der, radio_der = centro_y_radio(self._INDICES_OJO_DER)

            self._centro_ojo_izq = centro_izq
            self._centro_ojo_der = centro_der
            self._radio_ojo_izq = radio_izq
            self._radio_ojo_der = radio_der
            self._ultimo_timestamp_rostro = timestamp

        except Exception as e:
            logger.warning(f"No se pudieron actualizar centros de ojos para frote: {e}")

    def _dedo_en_ojo(
        self,
        punto: Tuple[int, int],
        ojo_izquierdo: bool,
    ) -> bool:
        """Devuelve True si el dedo (x, y) está dentro del círculo definido alrededor del ojo indicado.
        """
        if ojo_izquierdo:
            if self._centro_ojo_izq is None or self._radio_ojo_izq is None:
                return False
            cx, cy = self._centro_ojo_izq
            r = self._radio_ojo_izq
        else:
            if self._centro_ojo_der is None or self._radio_ojo_der is None:
                return False
            cx, cy = self._centro_ojo_der
            r = self._radio_ojo_der

        x, y = punto
        dx = float(x) - cx
        dy = float(y) - cy
        dist = np.hypot(dx, dy)
        return dist <= r

    def _reset_estado(self) -> None:
        """Resetea parcialmente el estado de frote (sin borrar info de ojos).
        """
        self._frote_activo = False
        self._duracion_frote_actual = 0.0
        self._frames_dedo_cerca_consecutivos = 0

    # ---------- Debug visual ----------

    def dibujar_debug_sobre_mascara(self, mascara: np.ndarray) -> np.ndarray:

        salida = mascara.copy()

        # Ojos
        color_ojo = (192, 255, 48)
        if self._centro_ojo_izq is not None and self._radio_ojo_izq is not None:
            cx, cy = int(self._centro_ojo_izq[0]), int(self._centro_ojo_izq[1])
            cv2.circle(salida, (cx, cy), int(self._radio_ojo_izq), color_ojo, 1)
            cv2.circle(salida, (cx, cy), 3, color_ojo, -1)
        if self._centro_ojo_der is not None and self._radio_ojo_der is not None:
            cx, cy = int(self._centro_ojo_der[0]), int(self._centro_ojo_der[1])
            cv2.circle(salida, (cx, cy), int(self._radio_ojo_der), color_ojo, 1)
            cv2.circle(salida, (cx, cy), 3, color_ojo, -1)

        # Puntas de dedos
        for (x, y) in self._ultimas_puntas_dedos_px:
            cv2.circle(salida, (x, y), 4, (0, 0, 255), -1)

        return salida

    def liberar(self) -> None:
        """Libera recursos de MediaPipe Hands."""
        if hasattr(self, "_hands") and self._hands is not None:
            self._hands.close()
            self._hands = None
            logger.info("Recursos de MediaPipe Hands liberados (frote ojos).")

    def __del__(self):
        try:
            self.liberar()
        except Exception:
            pass
