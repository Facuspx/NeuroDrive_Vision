"""
Módulo de captura de video optimizado para Raspberry Pi 5.
Maneja cámaras USB, CSI (Raspberry Pi Camera) y archivos de video.
"""

import cv2
import logging
from typing import Optional, Tuple
from enum import Enum

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorCapturaVideo(Exception):
    """
    Excepción específica para errores en la captura de video.
    
    Attributes
    ----------
    mensaje : str
        Descripción del error.
    codigo_error : str | None
        Código identificador del tipo de error (opcional).
    """
    
    def __init__(self, mensaje: str, codigo_error: Optional[str] = None):
        """
        Parameters
        ----------
        mensaje : str
            Descripción detallada del error.
        codigo_error : str | None
            Código de error para categorización (ej: 'CAM_NO_DISPONIBLE').
        """
        self.mensaje = mensaje
        self.codigo_error = codigo_error
        super().__init__(self.mensaje)
    
    def __str__(self) -> str:
        """Representación en string del error."""
        if self.codigo_error:
            return f"[{self.codigo_error}] {self.mensaje}"
        return self.mensaje


class TipoFuente(Enum):
    """Tipos de fuente de video soportados."""
    CAMARA_USB = "usb"
    CAMARA_CSI = "csi"  # Raspberry Pi Camera Module
    ARCHIVO = "archivo"


class CapturadorVideo:
    """
    Clase responsable de manejar la fuente de video (cámara web o archivo).
    
    Características:
    - Compatible con Raspberry Pi 5 (USB y CSI cameras)
    - Manejo robusto de errores
    - Reintentos automáticos
    - Validación de frames
    """
    
    # Constantes de configuración
    MAX_REINTENTOS = 3
    TIMEOUT_LECTURA = 5  # segundos
    
    def __init__(
        self,
        indice_camara: int = 0,
        ruta_video: Optional[str] = None,
        resolucion: Optional[Tuple[int, int]] = None,
        usar_csi: bool = False,
        fps_deseado: Optional[int] = None
    ) -> None:
        """
        Parámetros
        ----------
        indice_camara : int
            Índice de la cámara a usar (0 suele ser la webcam principal).
        ruta_video : str | None
            Ruta a un archivo de video. Si se especifica, tiene prioridad sobre la cámara.
        resolucion : (ancho, alto) | None
            Resolución deseada (ej: (640, 480) para mejor rendimiento en RPi).
        usar_csi : bool
            Si True, intenta usar Raspberry Pi Camera Module (CSI).
        fps_deseado : int | None
            FPS objetivo para la captura (útil para limitar carga en RPi).
        """
        self.indice_camara = indice_camara
        self.ruta_video = ruta_video
        self.resolucion = resolucion or (640, 480)  # Resolución por defecto para RPi
        self.fps_deseado = fps_deseado
        self.usar_csi = usar_csi
        
        self._captura: Optional[cv2.VideoCapture] = None
        self._tipo_fuente: TipoFuente = self._determinar_tipo_fuente()
        self._frames_leidos: int = 0
        self._frames_fallidos: int = 0
        
    def _determinar_tipo_fuente(self) -> TipoFuente:
        """Determina el tipo de fuente de video a usar."""
        if self.ruta_video is not None:
            return TipoFuente.ARCHIVO
        elif self.usar_csi:
            return TipoFuente.CAMARA_CSI
        else:
            return TipoFuente.CAMARA_USB
    
    def iniciar(self) -> None:
        """
        Abre la cámara o el archivo de video con reintentos automáticos.
        
        Raises
        ------
        ErrorCapturaVideo
            Si no se puede abrir la fuente después de MAX_REINTENTOS.
        """
        for intento in range(self.MAX_REINTENTOS):
            try:
                self._intentar_abrir_fuente()
                
                if not self._captura or not self._captura.isOpened():
                    raise ErrorCapturaVideo("No se pudo abrir la fuente")
                
                # Configurar propiedades
                self._configurar_captura()
                
                # Validar que podemos leer frames
                if not self._validar_lectura_inicial():
                    raise ErrorCapturaVideo("No se puede leer frames de la fuente")
                
                logger.info(f"Fuente de video iniciada correctamente ({self._tipo_fuente.value})")
                logger.info(f"Resolución: {self.obtener_resolucion()}, FPS: {self.obtener_fps():.2f}")
                return
                
            except Exception as e:
                logger.warning(f"Intento {intento + 1}/{self.MAX_REINTENTOS} falló: {e}")
                self.liberar()
                
                if intento == self.MAX_REINTENTOS - 1:
                    raise ErrorCapturaVideo(
                        f"No se pudo abrir la fuente de video después de {self.MAX_REINTENTOS} intentos: {e}"
                    )
    
    def _intentar_abrir_fuente(self) -> None:
        """Intenta abrir la fuente de video según su tipo."""
        if self._tipo_fuente == TipoFuente.ARCHIVO:
            self._captura = cv2.VideoCapture(self.ruta_video)
            
        elif self._tipo_fuente == TipoFuente.CAMARA_CSI:
            # Para Raspberry Pi Camera Module (CSI)
            # Usando GStreamer pipeline para mejor rendimiento
            gst_pipeline = (
                f"libcamerasrc ! "
                f"video/x-raw,width={self.resolucion[0]},height={self.resolucion[1]},framerate=30/1 ! "
                f"videoconvert ! appsink"
            )
            self._captura = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            
        else:  # CAMARA_USB
            # Para Raspberry Pi 5, V4L2 suele funcionar mejor que DSHOW
            self._captura = cv2.VideoCapture(self.indice_camara, cv2.CAP_V4L2)
            
            # Fallback a backend por defecto si V4L2 falla
            if not self._captura.isOpened():
                logger.warning("V4L2 falló, intentando con backend por defecto")
                self._captura = cv2.VideoCapture(self.indice_camara)
    
    def _configurar_captura(self) -> None:
        """Configura las propiedades de la captura."""
        if self._captura is None:
            return
        
        # Configurar resolución
        ancho, alto = self.resolucion
        self._captura.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
        self._captura.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)
        
        # Configurar FPS si se especificó
        if self.fps_deseado is not None:
            self._captura.set(cv2.CAP_PROP_FPS, self.fps_deseado)
        
        # Configuraciones adicionales para optimizar rendimiento en RPi
        if self._tipo_fuente == TipoFuente.CAMARA_USB:
            # Desactivar autoenfoque (puede causar lag)
            self._captura.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            
            # Configurar buffer size (reducir para menor latencia)
            self._captura.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    def _validar_lectura_inicial(self) -> bool:
        """
        Valida que se pueden leer frames de la fuente.
        
        Returns
        -------
        bool
            True si se pudo leer al menos un frame válido.
        """
        for _ in range(3):  # Intentar leer 3 frames
            ret, frame = self._captura.read()
            if ret and frame is not None and frame.size > 0:
                return True
        return False
    
    def leer_frame(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """
        Lee un frame de la fuente de video con validación.
        
        Returns
        -------
        ok : bool
            Indica si la lectura fue exitosa.
        frame : ndarray | None
            Imagen BGR obtenida. Es None si ok es False.
        
        Raises
        ------
        ErrorCapturaVideo
            Si la captura no fue inicializada.
        """
        if self._captura is None:
            raise ErrorCapturaVideo(
                "La captura de video no fue inicializada. Llama a 'iniciar()' primero."
            )
        
        ok, frame = self._captura.read()
        
        if not ok:
            self._frames_fallidos += 1
            logger.debug(f"Fallo al leer frame (total fallidos: {self._frames_fallidos})")
            return False, None
        
        # Validar que el frame no está vacío o corrupto
        if frame is None or frame.size == 0:
            self._frames_fallidos += 1
            logger.warning("Frame vacío o corrupto recibido")
            return False, None
        
        self._frames_leidos += 1
        return True, frame
    
    def obtener_fps(self) -> float:
        """
        Devuelve los FPS reportados por la fuente de video.
        
        Returns
        -------
        float
            FPS de la fuente (puede ser 0 si no está disponible).
        
        Raises
        ------
        ErrorCapturaVideo
            Si la captura no fue inicializada.
        """
        if self._captura is None:
            raise ErrorCapturaVideo("La captura de video no fue inicializada.")
        
        fps = float(self._captura.get(cv2.CAP_PROP_FPS))
        
        # Algunos dispositivos retornan 0, usar un valor por defecto razonable
        if fps == 0:
            logger.warning("FPS no disponible desde la fuente, usando 30 por defecto")
            fps = 30.0
        
        return fps
    
    def obtener_resolucion(self) -> Tuple[int, int]:
        """
        Devuelve la resolución actual (ancho, alto) de la captura.
        
        Returns
        -------
        Tuple[int, int]
            (ancho, alto) en píxeles.
        
        Raises
        ------
        ErrorCapturaVideo
            Si la captura no fue inicializada.
        """
        if self._captura is None:
            raise ErrorCapturaVideo("La captura de video no fue inicializada.")
        
        ancho = int(self._captura.get(cv2.CAP_PROP_FRAME_WIDTH))
        alto = int(self._captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        return ancho, alto
    
    def obtener_estadisticas(self) -> dict:
        """
        Retorna estadísticas de la captura.
        
        Returns
        -------
        dict
            Diccionario con estadísticas de frames.
        """
        total = self._frames_leidos + self._frames_fallidos
        tasa_exito = (self._frames_leidos / total * 100) if total > 0 else 0
        
        return {
            "frames_leidos": self._frames_leidos,
            "frames_fallidos": self._frames_fallidos,
            "tasa_exito": tasa_exito,
            "tipo_fuente": self._tipo_fuente.value
        }
    
    def reiniciar(self) -> None:
        """
        Reinicia la captura de video.
        Útil si la fuente se desconecta o hay problemas.
        """
        logger.info("Reiniciando captura de video...")
        self.liberar()
        self.iniciar()
    
    def liberar(self) -> None:
        """Libera la cámara / archivo de video y recursos asociados."""
        if self._captura is not None:
            self._captura.release()
            self._captura = None
            logger.info("Recursos de captura liberados")
    
    def __enter__(self):
        """Soporte para context manager."""
        self.iniciar()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Libera recursos al salir del context manager."""
        self.liberar()
        return False
    
    def __del__(self):
        """Asegura que los recursos se liberen al destruir el objeto."""
        self.liberar()