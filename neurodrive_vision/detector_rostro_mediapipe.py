from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field #más claro para objetos que solo almacenan datos.
from typing import List, Tuple, Optional, Dict
from abc import ABC, abstractmethod

import cv2
import numpy as np

# Importación condicional de MediaPipe 
try:
    import mediapipe as mp
    MEDIAPIPE_DISPONIBLE = True
except ImportError:
    MEDIAPIPE_DISPONIBLE = False
    mp = None

logger = logging.getLogger(__name__)


class ErrorDetectorRostro(Exception):                               #   Excepción base para errores en detección de rostro
    
    def __init__(self, mensaje: str, codigo_error: Optional[str] = None):
        self.mensaje = mensaje
        self.codigo_error = codigo_error
        super().__init__(self.mensaje)
    
    def __str__(self) -> str:
        if self.codigo_error:
            return f"[{self.codigo_error}] {self.mensaje}"
        return self.mensaje


class ErrorInicializacionDetector(ErrorDetectorRostro):
    
    def __init__(self, mensaje: str):
        super().__init__(mensaje, codigo_error="INIT_DETECTOR_ERROR")


class ErrorProcesamientoFrame(ErrorDetectorRostro):
    
    def __init__(self, mensaje: str):
        super().__init__(mensaje, codigo_error="PROCESS_FRAME_ERROR")


# ==============================
#   Estructuras de datos
# ==============================

@dataclass
class DatosRostro:

    rostro_presente: bool
    puntos_normalizados: Optional[List[Tuple[float, float, float]]] = None
    puntos_pixeles: Optional[List[Tuple[int, int]]] = None
    resolucion: Optional[Tuple[int, int]] = None
    confiabilidad: float = 0.0
    timestamp: float = field(default_factory=time.time)
    tiempo_procesamiento: float = 0.0


@dataclass
class MetricasDetector:             #Métricas de rendimiento del detector para monitorear el comportamiento en Raspberry Pi.
                                    #Si todo va bien lo podemos borrar
    frames_procesados: int = 0
    frames_con_rostro: int = 0
    frames_sin_rostro: int = 0
    errores_procesamiento: int = 0
    tiempo_total_procesamiento: float = 0.0
    tiempo_promedio_frame: float = 0.0
    fps_promedio: float = 0.0
    
    def actualizar(self, datos_rostro: DatosRostro, error: bool = False) -> None:
        """Actualiza las métricas con el resultado de un frame."""
        self.frames_procesados += 1
        
        if error:
            self.errores_procesamiento += 1
            return
        
        if datos_rostro.rostro_presente:
            self.frames_con_rostro += 1
        else:
            self.frames_sin_rostro += 1
        
        self.tiempo_total_procesamiento += datos_rostro.tiempo_procesamiento
        self.tiempo_promedio_frame = (
            self.tiempo_total_procesamiento / self.frames_procesados
        )
        
        if self.tiempo_promedio_frame > 0:
            self.fps_promedio = 1.0 / self.tiempo_promedio_frame
    
    def obtener_reporte(self) -> Dict[str, float]:
        """Retorna un diccionario con las métricas."""
        tasa_deteccion = (
            (self.frames_con_rostro / self.frames_procesados * 100)
            if self.frames_procesados > 0 else 0.0
        )
        
        return {
            "frames_procesados": self.frames_procesados,
            "frames_con_rostro": self.frames_con_rostro,
            "frames_sin_rostro": self.frames_sin_rostro,
            "errores": self.errores_procesamiento,
            "tasa_deteccion_pct": round(tasa_deteccion, 2),
            "tiempo_promedio_ms": round(self.tiempo_promedio_frame * 1000, 2),
            "fps_promedio": round(self.fps_promedio, 2),
        }
    
    def reiniciar(self) -> None:
        """Reinicia todas las métricas a cero."""
        self.__init__()


# ==============================
#   Clase base (interfaz genérica)
# ==============================

class DetectorRostroBase(ABC):

    @abstractmethod
    def procesar_frame(self, frame_bgr: np.ndarray) -> DatosRostro:
        ...

    @abstractmethod
    def obtener_metricas(self) -> MetricasDetector:
        """Retorna las métricas de rendimiento del detector."""
        ...

    @abstractmethod
    def reiniciar_metricas(self) -> None:
        """Reinicia las métricas de rendimiento."""
        ...

    @abstractmethod
    def liberar(self) -> None:
        """Libera recursos asociados al detector."""
        ...


# ==============================
#   Implementación con MediaPipe
# ==============================

class DetectorRostroMediaPipe(DetectorRostroBase):

    def __init__(
        self,
        max_rostros: int = 1,
        confianza_minima_deteccion: float = 0.6,
        confianza_minima_seguimiento: float = 0.6,
        refinar_contornos: bool = True,
        modelo_complejidad: int = 1,
        habilitar_cache: bool = True,
        max_frames_sin_deteccion: int = 5
    ) -> None:

        if not MEDIAPIPE_DISPONIBLE:
            raise ErrorInicializacionDetector(
                "MediaPipe no está instalado. "
                "Instala con: pip install mediapipe"
            )
        
        self._max_rostros = max_rostros
        self._habilitar_cache = habilitar_cache
        self._max_frames_sin_deteccion = max_frames_sin_deteccion
        
        # Caché para estabilidad
        self._ultimo_resultado: Optional[DatosRostro] = None
        self._frames_consecutivos_sin_rostro: int = 0
        
        # Métricas
        self._metricas = MetricasDetector()
        
        try:
            self._mp_face_mesh = mp.solutions.face_mesh
            
            # Configuración optimizada para Raspberry Pi
            self._face_mesh = self._mp_face_mesh.FaceMesh(
                static_image_mode=False,  # Video mode (más eficiente)
                max_num_faces=max_rostros,
                refine_landmarks=refinar_contornos,
                min_detection_confidence=confianza_minima_deteccion,
                min_tracking_confidence=confianza_minima_seguimiento,
                # Nota: model_complexity no es parámetro oficial de MediaPipe
                # pero se puede configurar internamente si se necesita
            )
            
            # Utilidades de dibujo (para debug)
            self._mp_drawing = mp.solutions.drawing_utils
            self._mp_drawing_styles = mp.solutions.drawing_styles
            
            logger.info(
                f"DetectorRostroMediaPipe inicializado correctamente "
                f"(max_rostros={max_rostros}, refine_landmarks={refinar_contornos}, "
                f"cache={habilitar_cache})"
            )
            
        except Exception as e:
            raise ErrorInicializacionDetector(
                f"Error al inicializar MediaPipe FaceMesh: {e}"
            )

    def procesar_frame(self, frame_bgr: np.ndarray) -> DatosRostro:

        tiempo_inicio = time.perf_counter()
        
        # Validación temprana
        if frame_bgr is None or frame_bgr.size == 0:
            logger.warning("Frame vacío o None recibido")
            resultado = self._crear_resultado_vacio(None)
            self._metricas.actualizar(resultado)
            return resultado
        
        alto, ancho = frame_bgr.shape[:2]
        resolucion = (ancho, alto)
        
        try:
            # MediaPipe requiere RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            
            # Optimización: marcar como no-escribible
            frame_rgb.flags.writeable = False
            
            # Procesar con MediaPipe
            resultados = self._face_mesh.process(frame_rgb)
            
            # Calcular tiempo de procesamiento
            tiempo_procesamiento = time.perf_counter() - tiempo_inicio
            
            # Procesar resultados
            if not resultados.multi_face_landmarks:
                # No se detectó rostro
                self._frames_consecutivos_sin_rostro += 1
                
                # Usar caché si está habilitado y no han pasado muchos frames
                if (self._habilitar_cache and 
                    self._ultimo_resultado is not None and
                    self._frames_consecutivos_sin_rostro <= self._max_frames_sin_deteccion):
                    
                    logger.debug(
                        f"Usando resultado en caché "
                        f"(frames sin detección: {self._frames_consecutivos_sin_rostro})"
                    )
                    
                    # Actualizar timestamp del resultado cacheado
                    resultado = DatosRostro(
                        rostro_presente=True,
                        puntos_normalizados=self._ultimo_resultado.puntos_normalizados,
                        puntos_pixeles=self._ultimo_resultado.puntos_pixeles,
                        resolucion=resolucion,
                        confiabilidad=0.8,  # Reducir confianza por ser caché
                        timestamp=time.time(),
                        tiempo_procesamiento=tiempo_procesamiento
                    )
                else:
                    # Sin caché o caché expirado
                    if self._frames_consecutivos_sin_rostro > self._max_frames_sin_deteccion:
                        self._ultimo_resultado = None  # Invalidar caché
                    
                    resultado = DatosRostro(
                        rostro_presente=False,
                        puntos_normalizados=None,
                        puntos_pixeles=None,
                        resolucion=resolucion,
                        confiabilidad=0.0,
                        timestamp=time.time(),
                        tiempo_procesamiento=tiempo_procesamiento
                    )
                
                self._metricas.actualizar(resultado)
                return resultado
            
            # Rostro detectado exitosamente
            self._frames_consecutivos_sin_rostro = 0
            
            # Extraer landmarks del primer rostro
            landmarks_rostro = resultados.multi_face_landmarks[0].landmark
            
            # Convertir a listas de tuplas
            puntos_normalizados: List[Tuple[float, float, float]] = []
            puntos_pixeles: List[Tuple[int, int]] = []
            
            for punto in landmarks_rostro:
                x_norm = float(punto.x)
                y_norm = float(punto.y)
                z_norm = float(punto.z)
                
                puntos_normalizados.append((x_norm, y_norm, z_norm))
                
                # Clipping para evitar puntos fuera del frame
                x_px = max(0, min(ancho - 1, int(x_norm * ancho)))
                y_px = max(0, min(alto - 1, int(y_norm * alto)))
                puntos_pixeles.append((x_px, y_px))
            
            resultado = DatosRostro(
                rostro_presente=True,
                puntos_normalizados=puntos_normalizados,
                puntos_pixeles=puntos_pixeles,
                resolucion=resolucion,
                confiabilidad=1.0,
                timestamp=time.time(),
                tiempo_procesamiento=tiempo_procesamiento
            )
            
            # Guardar en caché
            if self._habilitar_cache:
                self._ultimo_resultado = resultado
            
            self._metricas.actualizar(resultado)
            return resultado
            
        except cv2.error as e:
            logger.error(f"Error de OpenCV al procesar frame: {e}")
            resultado = self._crear_resultado_vacio(resolucion)
            self._metricas.actualizar(resultado, error=True)
            return resultado
            
        except Exception as e:
            logger.error(f"Error inesperado al procesar frame: {e}", exc_info=True)
            resultado = self._crear_resultado_vacio(resolucion)
            self._metricas.actualizar(resultado, error=True)
            return resultado

    def _crear_resultado_vacio(self, resolucion: Optional[Tuple[int, int]]) -> DatosRostro:
        """Helper para crear un DatosRostro vacío."""
        return DatosRostro(
            rostro_presente=False,
            puntos_normalizados=None,
            puntos_pixeles=None,
            resolucion=resolucion,
            confiabilidad=0.0,
            timestamp=time.time(),
            tiempo_procesamiento=0.0
        )

    def dibujar_malla(
        self,
        frame_bgr: np.ndarray,
        datos_rostro: DatosRostro,
        dibujar_contornos: bool = False,
        dibujar_puntos: bool = True,
        grosor_linea: int = 1,
        color_contorno: Tuple[int, int, int] = (255, 255, 0)
    ) -> np.ndarray:

        # Creamos una imagen negra del mismo tamaño que el frame original
        mascara = np.zeros_like(frame_bgr)

        if not datos_rostro.rostro_presente:
            # No hay rostro -> devolvemos solo fondo negro
            return mascara

        if dibujar_puntos and datos_rostro.puntos_pixeles is not None:
            for (x_px, y_px) in datos_rostro.puntos_pixeles:
                cv2.circle(mascara, (x_px, y_px), 1, color_contorno, -1)

        return mascara


    def obtener_metricas(self) -> MetricasDetector:
        """Retorna las métricas actuales del detector."""
        return self._metricas

    def reiniciar_metricas(self) -> None:
        """Reinicia las métricas de rendimiento a cero."""
        self._metricas.reiniciar()
        logger.info("Métricas del detector reiniciadas")

    def invalidar_cache(self) -> None:
        """Invalida el caché de resultados manualmente."""
        self._ultimo_resultado = None
        self._frames_consecutivos_sin_rostro = 0
        logger.debug("Caché de resultados invalidado")

    def liberar(self) -> None:
        """Libera recursos de MediaPipe FaceMesh."""
        if hasattr(self, '_face_mesh') and self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None
            logger.info("Recursos de MediaPipe FaceMesh liberados")

    def __del__(self):
        """Garantiza la liberación de recursos al destruir el objeto."""
        self.liberar()

    def __enter__(self):
        """Soporte para context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Libera recursos al salir del context manager."""
        self.liberar()
        return False