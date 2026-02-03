from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List
from .contador_eventos import SalidaEventos
import logging

logger = logging.getLogger(__name__)


# ==============================
#   Definiciones de estados
# ==============================

class EstadoAlerta(Enum):
    """Estados de alto nivel de la máquina de somnolencia"""
    NORMAL = auto()              # Conductor atento / patrón normal
    PRE_SOMNOLENCIA = auto()     # Primeras señales (bostezos, parpadeos altos, atención media)
    SOMNOLENCIA_MEDIA = auto()   # Microsueños aislados / patrón preocupante
    SOMNOLENCIA_ALTA = auto()    # Microsueños repetidos / peligro alto
    MODO_DEGRADADO = auto()      # Fallos de sensores / demasiadas ventanas no confiables


@dataclass
class SalidaMaquinaEstados:
    """Salida compacta de la máquina de estados para el resto del sistema"""

    estado_alerta: EstadoAlerta
    nivel_alerta: int  # 0 = normal, 1 = leve, 2 = media, 3 = alta

    # Acciones sugeridas (la capa superior decidirá cómo accionar físicamente)
    activar_alarma_visual: bool = False
    activar_alarma_sonora: bool = False
    activar_vibracion_pulsera: bool = False
    enviar_notificacion_supervisor: bool = False

    # Robustez / calidad
    modo_degradado: bool = False
    motivo_modo_degradado: str = ""

    # Métricas auxiliares
    ventanas_no_confiables_recientes: int = 0

    # Para logging / debug en máscara
    mensaje_debug: str = ""


# ==============================
#   Clase integradora
# ==============================

class IntegradorMaquinaEstados:

    def __init__(
        self,
        # Ventanas de tiempo (segundos)
        ventana_eventos_corta: float = 60.0,   # análisis de eventos recientes ~1 min
        ventana_eventos_larga: float = 300.0,  # análisis de tendencia ~5 min

        # Umbrales de decisión (provisionales, se ajustan con pruebas)
        max_microsuenos_en_corta: int = 1,
        max_microsuenos_en_larga: int = 3,
        max_bostezos_en_corta: int = 3,
        max_cabeceos_en_corta: int = 1,

        max_ventanas_no_confiables_en_corta: int = 5,
    ) -> None:
        # Estado actual
        self._estado_actual: EstadoAlerta = EstadoAlerta.NORMAL

        # Historial de eventos con timestamps
        self._hist_microsuenos: List[float] = []
        self._hist_bostezos: List[float] = []
        self._hist_cabeceos: List[float] = []
        self._hist_ventanas_no_confiables: List[float] = []

        # Configuración
        self._ventana_corta = ventana_eventos_corta
        self._ventana_larga = ventana_eventos_larga

        self._max_microsuenos_corta = max_microsuenos_en_corta
        self._max_microsuenos_larga = max_microsuenos_en_larga
        self._max_bostezos_corta = max_bostezos_en_corta
        self._max_cabeceos_corta = max_cabeceos_en_corta

        self._max_ventanas_no_confiables_corta = max_ventanas_no_confiables_en_corta

        logger.info("IntegradorMaquinaEstados inicializado.")

    # ---------- API principal ----------

    def actualizar(self, salida_eventos: SalidaEventos) -> SalidaMaquinaEstados:

        t = salida_eventos.timestamp
        ev = salida_eventos.eventos

        # 1) Actualizar historiales
        self._actualizar_historiales(t, salida_eventos)

        # 2) Podar historiales según ventanas de tiempo
        self._podar_historiales(t)

        # 3) Evaluar nuevo estado según:
        #    - eventos recientes
        #    - atención estimada
        #    - ventanas no confiables
        nuevo_estado, mensaje = self._evaluar_estado(t, salida_eventos)

        self._estado_actual = nuevo_estado

        # 4) Traducir estado a salidas (alertas/acciones)
        salida_maquina = self._generar_salida(t, salida_eventos, mensaje)

        return salida_maquina

    # ---------- Historiales ----------

    def _actualizar_historiales(self, timestamp: float, salida: SalidaEventos) -> None:
        """ Registra eventos discretos en listas con timestamps para evaluar tasas dentro de ventanas de tiempo.
        """
        ev = salida.eventos

        if ev.microsueno:
            self._hist_microsuenos.append(timestamp)

        if ev.bostezo:
            self._hist_bostezos.append(timestamp)

        if ev.cabeceo:
            self._hist_cabeceos.append(timestamp)

        # Solo consideramos ventanas no confiables que NO sean por frote de ojos
        # para el modo degradado. El frote es un comportamiento normal del conductor.
        if salida.ventana_no_confiable and not salida.frote_activo:
            self._hist_ventanas_no_confiables.append(timestamp)


    def _podar_lista(self, lista: List[float], timestamp: float, ventana: float) -> None:
        """ Elimina de la lista los eventos cuya marca de tiempo esté fuera
        de la ventana [timestamp - ventana, timestamp].
        """
        limite = timestamp - ventana
        while lista and lista[0] < limite:
            lista.pop(0)

    def _podar_historiales(self, timestamp: float) -> None:
        """Podamos todas las listas de eventos según la ventana larga.
        (la ventana corta se evalúa usando subintervalos).
        """
        self._podar_lista(self._hist_microsuenos, timestamp, self._ventana_larga)
        self._podar_lista(self._hist_bostezos, timestamp, self._ventana_larga)
        self._podar_lista(self._hist_cabeceos, timestamp, self._ventana_larga)
        self._podar_lista(self._hist_ventanas_no_confiables, timestamp, self._ventana_larga)

    # ---------- Evaluación de estado ----------

    def _contar_en_ventana(self, lista: List[float], timestamp: float, ventana: float) -> int:
        """Cuenta cuántos eventos hay en la subventana [t - ventana, t].
        """
        limite = timestamp - ventana
        return sum(1 for t in lista if t >= limite)

    def _evaluar_estado(self, timestamp: float, salida: SalidaEventos) -> tuple[EstadoAlerta, str]:
        """Evalúa el nuevo estado de alerta según:
        - tasa de microsueños / bostezos / cabeceos
        - estimación de atención
        - calidad de las ventanas (ventanas no confiables)
        """
        estado = self._estado_actual
        mensaje = ""

        # Contadores en ventana corta
        microsuenos_corta = self._contar_en_ventana(self._hist_microsuenos, timestamp, self._ventana_corta)
        microsuenos_larga = len(self._hist_microsuenos)  # ya podado a ventana larga
        bostezos_corta = self._contar_en_ventana(self._hist_bostezos, timestamp, self._ventana_corta)
        cabeceos_corta = self._contar_en_ventana(self._hist_cabeceos, timestamp, self._ventana_corta)
        ventanas_no_conf_corta = self._contar_en_ventana(
            self._hist_ventanas_no_confiables, timestamp, self._ventana_corta
        )

        atencion = salida.atencion

        # 1) Verificar modo degradado (calidad de señal muy mala)
        if ventanas_no_conf_corta >= self._max_ventanas_no_confiables_corta:
            mensaje = (
                f"Muchas ventanas no confiables en {self._ventana_corta}s "
                f"({ventanas_no_conf_corta}), activando modo degradado."
            )
            return EstadoAlerta.MODO_DEGRADADO, mensaje

        # 2) Somnolencia alta: microsueños repetidos
        if microsuenos_larga >= self._max_microsuenos_larga:
            mensaje = (
                f"{microsuenos_larga} microsuenos en ~{self._ventana_larga/60:.1f} min. "
                "somnolencia alta."
            )
            return EstadoAlerta.SOMNOLENCIA_ALTA, mensaje

        if microsuenos_corta > self._max_microsuenos_corta:
            mensaje = (
                f"{microsuenos_corta} microsuenos en {self._ventana_corta}s. "
                "somnolencia media."
            )
            return EstadoAlerta.SOMNOLENCIA_MEDIA, mensaje

        # 3) Bostezos y cabeceos en ventana corta -> pre-somnolencia
        if bostezos_corta >= self._max_bostezos_corta or cabeceos_corta >= self._max_cabeceos_corta:
            mensaje = (
                f"Bostezos={bostezos_corta}, Cabeceos={cabeceos_corta} en {self._ventana_corta}s. "
                "pre-somnolencia."
            )
            return EstadoAlerta.PRE_SOMNOLENCIA, mensaje

        # 4) Atención estimada "media" o "baja" sin eventos fuertes
        if atencion.categoria == "baja":
            mensaje = f"Atencion baja segun contador_eventos: {atencion.motivo}"
            return EstadoAlerta.PRE_SOMNOLENCIA, mensaje

        # Atención media sola NO alcanza para sacar de NORMAL.
        # Solo si ya estamos en PRE_SOMNOLENCIA, la atención media ayuda a mantenernos ahí.
        if atencion.categoria == "media" and estado == EstadoAlerta.PRE_SOMNOLENCIA:
            mensaje = f"Atencion media mantiene estado de pre-somnolencia: {atencion.motivo}"
            return EstadoAlerta.PRE_SOMNOLENCIA, mensaje


        # 5) Caso por defecto: estado normal
        mensaje = "Patron dentro de rangos normales."
        return EstadoAlerta.NORMAL, mensaje

    # ---------- Generación de acciones ----------

    def _generar_salida(
        self,
        timestamp: float,
        salida_eventos: SalidaEventos,
        mensaje_estado: str,
    ) -> SalidaMaquinaEstados:
        """ Traduce el estado actual de la máquina a nivel de alerta y acciones sugeridas"""
        estado = self._estado_actual

        if estado == EstadoAlerta.NORMAL:
            nivel = 0
            activar_visual = False
            activar_sonora = False
            activar_vibracion = False
            notificar = False

        elif estado == EstadoAlerta.PRE_SOMNOLENCIA:
            nivel = 1
            activar_visual = True
            activar_sonora = False
            activar_vibracion = False
            notificar = False

        elif estado == EstadoAlerta.SOMNOLENCIA_MEDIA:
            nivel = 2
            activar_visual = True
            activar_sonora = True
            activar_vibracion = True
            notificar = False

        elif estado == EstadoAlerta.SOMNOLENCIA_ALTA:
            nivel = 3
            activar_visual = True
            activar_sonora = True
            activar_vibracion = True
            notificar = True

        elif estado == EstadoAlerta.MODO_DEGRADADO:
            nivel = 0  # No podemos afirmar alarma por somnolencia
            activar_visual = True
            activar_sonora = False
            activar_vibracion = False
            notificar = True  # informar fallo de sistema

        else:
            # Fallback
            nivel = 0
            activar_visual = False
            activar_sonora = False
            activar_vibracion = False
            notificar = False

        ventanas_no_conf_corta = self._contar_en_ventana(
            self._hist_ventanas_no_confiables, timestamp, self._ventana_corta
        )

        modo_degradado = (estado == EstadoAlerta.MODO_DEGRADADO)
        motivo_degradado = ""
        if modo_degradado:
            motivo_degradado = (
                f"Exceso de ventanas no confiables en {self._ventana_corta}s "
                f"({ventanas_no_conf_corta})."
            )

        mensaje_debug = (
            f"Estado={estado.name}, nivel={nivel}, msg='{mensaje_estado}', "
            f"no_conf_corta={ventanas_no_conf_corta}"
        )

        salida = SalidaMaquinaEstados(
            estado_alerta=estado,
            nivel_alerta=nivel,
            activar_alarma_visual=activar_visual,
            activar_alarma_sonora=activar_sonora,
            activar_vibracion_pulsera=activar_vibracion,
            enviar_notificacion_supervisor=notificar,
            modo_degradado=modo_degradado,
            motivo_modo_degradado=motivo_degradado,
            ventanas_no_confiables_recientes=ventanas_no_conf_corta,
            mensaje_debug=mensaje_debug,
        )

        return salida
