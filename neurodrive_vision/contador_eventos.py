from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List
from .medidas_rostro import MedidasRostro
from .detector_frote_ojos import ResultadoFroteOjos
import logging
logger = logging.getLogger(__name__)


# ==============================
#   Estructuras de datos
# ==============================

@dataclass
class EstadoOjos:
    estado: str = "desconocido"  # "abierto" | "cerrado" | "desconocido"
    duracion_estado: float = 0.0
    ear_actual: Optional[float] = None

@dataclass
class EventosSomnolencia:
    parpadeo: bool = False
    microsueno: bool = False
    bostezo: bool = False
    cabeceo: bool = False

@dataclass
class AtencionConductor:
    nivel: float = 1.0           # 0.0–1.0
    categoria: str = "alta"      # "alta" | "media" | "baja" | "indefinida"
    motivo: str = "normal"

@dataclass
class SalidaEventos:
    timestamp: float
    estado_ojos: EstadoOjos
    eventos: EventosSomnolencia
    atencion: AtencionConductor

    #calidad de ventana para el conteo o no de eventos
    ventana_no_confiable: bool = False
    frote_activo: bool = False


# ==============================
#   Clase principal
# ==============================

class ContadorEventosSomnolencia:
    """ Clase que acumula estado en el tiempo y genera eventos de somnolencia y una estimación básica de atención del conductor."""

    def __init__(
        self,
        umbral_ear_cerrado: float = 0.20,
        dur_min_parpadeo: float = 0.10,
        dur_max_parpadeo: float = 0.40,
        dur_min_microsueno: float = 1.5,
        umbral_mar_bostezo: float = 0.6,
        dur_min_bostezo: float = 1.0,
        ventana_interparpadeos_seg: float = 60.0,
        interparpadeo_atencion_baja: float = 8.0,
    ) -> None:

        # Umbrales y constantes
        self.umbral_ear_cerrado = umbral_ear_cerrado
        self.dur_min_parpadeo = dur_min_parpadeo
        self.dur_max_parpadeo = dur_max_parpadeo
        self.dur_min_microsueno = dur_min_microsueno
        self.umbral_mar_bostezo = umbral_mar_bostezo
        self.dur_min_bostezo = dur_min_bostezo
        self.ventana_interparpadeos_seg = ventana_interparpadeos_seg
        self.interparpadeo_atencion_baja = interparpadeo_atencion_baja

        # Estado interno global
        self._ultimo_timestamp: Optional[float] = None
        self._estado_ojos = EstadoOjos()

        # Para ojos (parpadeos / microsueños)
        self._conteo_parpadeos_total: int = 0
        self._conteo_microsuenos_total: int = 0
        self._ultimo_parpadeo_timestamp: Optional[float] = None
        self._historial_interparpadeos: List[float] = []

        # Suavizado de EAR
        self._ear_filtrado: Optional[float] = None
        self._alpha_ear: float = 0.5  # 0.0 = sin suavizar, 0.99 = muy suave

        # Hysteresis para ojo cerrado/abierto
        self._umbral_ear_cerrar: float = 0.18
        self._umbral_ear_abrir: float = 0.22

        # Período refractario entre parpadeos (seg)
        self._tiempo_refractario_parpadeo: float = 0.25

        # Para boca (bostezos)
        self._boca_abierta: bool = False
        self._tiempo_boca_abierta: float = 0.0
        self._conteo_bostezos_total: int = 0

        # Para cabeza (cabeceos simples)
        self._cabeceo_activo: bool = False
        self._tiempo_cabeza_abajo: float = 0.0
        self._dur_min_cabeceo: float = 0.8  # se podría parametrizar aqui era 1.0
        self._conteo_cabeceos_total: int = 0

        # Baseline para cabeza (se irá actualizando cuando no hay eventos raros)
        self._baseline_altura_nariz: Optional[float] = None
        self._baseline_altura_menton: Optional[float] = None

        # Para estímulos de pulsera (futuro)
        self._ultimo_estimulo: Optional[float] = None
        self._ultimo_tiempo_respuesta: Optional[float] = None
        self._latencias_respuesta: List[float] = []

    # ---------- API pública ----------

    def actualizar(
        self,
        timestamp: float,
        medidas: MedidasRostro,
        resultado_frote: Optional[ResultadoFroteOjos] = None,
    ) -> SalidaEventos:

        eventos = EventosSomnolencia()

        if self._ultimo_timestamp is None:
            dt = 0.0
        else:
            dt = max(0.0, timestamp - self._ultimo_timestamp)

        self._ultimo_timestamp = timestamp

        # --- Integración con frote de ojos (ventana no confiable) ---
        ventana_no_confiable = False
        frote_activo = False

        if resultado_frote is not None:
            frote_activo = resultado_frote.frote_activo
            if resultado_frote.frote_activo:
                ventana_no_confiable = True

        # Si la ventana es no confiable, NO queremos generar eventos de somnolencia,
        # pero sí queremos seguir actualizando estados y filtros internos.
        permite_eventos = not ventana_no_confiable

        # 1) Actualizar estado de ojos y detectar parpadeos / microsueños
        self._actualizar_ojos_y_eventos(dt, medidas, eventos, permite_eventos)

        # 2) Actualizar estado de boca (bostezos)
        self._actualizar_boca(dt, medidas, eventos, permite_eventos)

        # 3) Actualizar estado de cabeza (cabeceos simples)
        self._actualizar_cabeza(dt, medidas, eventos, permite_eventos)

        # 4) Estimar atención del conductor (versión 1)
        atencion = self._estimar_atencion(timestamp)

        # Si la ventana es no confiable por frote, degradamos la confianza de la atención
        if ventana_no_confiable:
            atencion.categoria = "indefinida"
            atencion.motivo = "Ventana no confiable (frote de ojos activo)"

        salida = SalidaEventos(
            timestamp=timestamp,
            estado_ojos=self._estado_ojos,
            eventos=eventos,
            atencion=atencion,
            ventana_no_confiable=ventana_no_confiable,
            frote_activo=frote_activo,
        )
        return salida

    def registrar_estimulo(self, timestamp: float) -> None:
        """ Registra que se envió un estímulo a la pulsera (ej. vibración + secuencia).
        Luego, al recibir la respuesta del conductor, se llamará a registrar_respuesta().
        """
        self._ultimo_estimulo = timestamp

    def registrar_respuesta(self, timestamp: float) -> None:
        """Registra que el conductor respondió a un estímulo anterior. Calcula latencia de respuesta y la almacena para usarla en el
        estimador de atención (por ahora, solo se guarda).
        """
        if self._ultimo_estimulo is None:
            logger.warning("Se registró una respuesta sin estímulo previo.")
            return

        latencia = max(0.0, timestamp - self._ultimo_estimulo)
        self._ultimo_tiempo_respuesta = timestamp
        self._latencias_respuesta.append(latencia)

        # Podríamos limitar la longitud de la lista para no crecer sin límite
        if len(self._latencias_respuesta) > 50:
            self._latencias_respuesta.pop(0)

        # Consumimos el estímulo
        self._ultimo_estimulo = None

    # ---------- Lógica interna: ojos ----------

    def _actualizar_ojos_y_eventos(
        self,
        dt: float,
        medidas: MedidasRostro,
        eventos: EventosSomnolencia,
        permite_eventos: bool,
    ) -> None:
        # Determinar EAR promedio
        ear_crudo = medidas.medidas_ojos.ear_promedio if medidas.medidas_ojos.valido else None

        if ear_crudo is None:
            # No actualizamos estado si no hay medida confiable
            self._estado_ojos.estado = "desconocido"
            self._estado_ojos.ear_actual = None
            return

        # 1) Suavizado exponencial del EAR
        if self._ear_filtrado is None:
            self._ear_filtrado = ear_crudo
        else:
            self._ear_filtrado = (
                self._alpha_ear * self._ear_filtrado +
                (1.0 - self._alpha_ear) * ear_crudo
            )

        ear = self._ear_filtrado
        self._estado_ojos.ear_actual = ear

        # 2) Hysteresis para determinar estado nuevo
        estado_actual = self._estado_ojos.estado

        if estado_actual in ("desconocido", "abierto"):
            # Solo cerramos si bajamos por debajo de umbral de cierre
            if ear < self._umbral_ear_cerrar:
                nuevo_estado = "cerrado"
            else:
                nuevo_estado = "abierto"
        else:  # estado_actual == "cerrado"
            # Solo abrimos si subimos por encima de umbral de apertura
            if ear > self._umbral_ear_abrir:
                nuevo_estado = "abierto"
            else:
                nuevo_estado = "cerrado"

        # 3) Actualizar duración de estado y detectar eventos al cambiar
        if estado_actual == nuevo_estado:
            self._estado_ojos.duracion_estado += dt
            return

        # Hay cambio de estado -> evaluamos el estado anterior
        dur_anterior = self._estado_ojos.duracion_estado
        estado_anterior = estado_actual

        if permite_eventos and estado_anterior == "cerrado":
            # Venimos de un período de ojos cerrados -> puede ser parpadeo o microsueño
            if self.dur_min_parpadeo <= dur_anterior <= self.dur_max_parpadeo:
                # Verificar período refractario
                if (
                    self._ultimo_parpadeo_timestamp is None or
                    (self._ultimo_timestamp - self._ultimo_parpadeo_timestamp) >= self._tiempo_refractario_parpadeo
                ):
                    eventos.parpadeo = True
                    self._conteo_parpadeos_total += 1
                    if self._ultimo_parpadeo_timestamp is not None:
                        inter = max(0.0, self._ultimo_timestamp - self._ultimo_parpadeo_timestamp)
                        self._agregar_interparpadeo(inter)
                    self._ultimo_parpadeo_timestamp = self._ultimo_timestamp

            elif dur_anterior >= self.dur_min_microsueno:
                eventos.microsueno = True
                self._conteo_microsuenos_total += 1

        # Reiniciar duración para el nuevo estado
        self._estado_ojos.estado = nuevo_estado
        self._estado_ojos.duracion_estado = dt

    def _agregar_interparpadeo(self, valor: float) -> None:
        """Agrega un nuevo inter-parpadeo al historial y poda según ventana de tiempo aproximada."""
        self._historial_interparpadeos.append(valor)
        # Podemos limitar la cantidad de muestras, por simplicidad
        if len(self._historial_interparpadeos) > 100:
            self._historial_interparpadeos.pop(0)

    # ---------- Lógica interna: boca ----------

    def _actualizar_boca(
        self,
        dt: float,
        medidas: MedidasRostro,
        eventos: EventosSomnolencia,
        permite_eventos: bool,
    ) -> None:
        if not medidas.medidas_boca.valido or medidas.medidas_boca.mar is None:
            # No tocamos el estado de boca si no hay datos
            return

        mar = medidas.medidas_boca.mar
        boca_abierta_ahora = mar >= self.umbral_mar_bostezo

        if boca_abierta_ahora:
            self._tiempo_boca_abierta += dt
            self._boca_abierta = True
        else:
            # Si se estaba abierta y se cerró, evaluamos duración
            if self._boca_abierta:
                if self._tiempo_boca_abierta >= self.dur_min_bostezo and permite_eventos:
                    eventos.bostezo = True
                    self._conteo_bostezos_total += 1
            self._boca_abierta = False
            self._tiempo_boca_abierta = 0.0

    # ---------- Lógica interna: cabeza (muy simple de momento) ----------

    def _actualizar_cabeza(
        self,
        dt: float,
        medidas: MedidasRostro,
        eventos: EventosSomnolencia,
        permite_eventos: bool,
    ) -> None:
        if not medidas.medidas_cabeza.valido:
            return

        altura_nariz = medidas.medidas_cabeza.altura_relativa_nariz or 0.0
        altura_menton = medidas.medidas_cabeza.altura_relativa_menton or 0.0

        # Inicializar baseline si aún no lo tenemos y parece postura "normal"
        if self._baseline_altura_nariz is None or self._baseline_altura_menton is None:
            self._baseline_altura_nariz = altura_nariz
            self._baseline_altura_menton = altura_menton
            return

        # Delta respecto al baseline (positivo = cabeza más abajo)
        delta_nariz = altura_nariz - self._baseline_altura_nariz
        delta_menton = altura_menton - self._baseline_altura_menton

        # Actualizar baseline suavemente cuando no parece haber cabeceo
        if not self._cabeceo_activo:
            alpha_base = 0.005
            self._baseline_altura_nariz = (
                (1 - alpha_base) * self._baseline_altura_nariz + alpha_base * altura_nariz
            )
            self._baseline_altura_menton = (
                (1 - alpha_base) * self._baseline_altura_menton + alpha_base * altura_menton
            )

        # Heuristic: cabeza significativamente abajo si ambos se movieron bastante hacia abajo
        umbral_delta = 0.03  # 10% de la altura de la imagen, se puede ajustar
        cabeza_abajo_ahora = (delta_nariz > umbral_delta) and (delta_menton > umbral_delta)

        if cabeza_abajo_ahora:
            self._tiempo_cabeza_abajo += dt
            if (
                self._tiempo_cabeza_abajo >= self._dur_min_cabeceo
                and not self._cabeceo_activo
                and permite_eventos
            ):
                eventos.cabeceo = True
                self._cabeceo_activo = True
                self._conteo_cabeceos_total += 1
        else:
            self._tiempo_cabeza_abajo = 0.0
            self._cabeceo_activo = False

    # ---------- Estimador de atención ----------

    def _estimar_atencion(self, timestamp: float) -> AtencionConductor:

        nivel = 1.0
        categoria = "alta"
        motivo = "patron de parpadeo normal"

        # 1) Somnolencia fuerte: microsueños recientes
        # De momento, usamos el conteo total como referencia (se puede refinar con ventanas de tiempo)
        if self._conteo_microsuenos_total > 0:
            nivel = 0.2
            categoria = "baja"
            motivo = "microsuenos detectados (somnolencia)"
            return AtencionConductor(nivel=nivel, categoria=categoria, motivo=motivo)

        # 2) Inter-parpadeos y mirada fija: posible mente en la nube
        if len(self._historial_interparpadeos) >= 3:
            promedio_inter = sum(self._historial_interparpadeos) / len(self._historial_interparpadeos)

            if promedio_inter > self.interparpadeo_atencion_baja:
                nivel = 0.5
                categoria = "media"
                motivo = (
                    f"inter-parpadeo promedio largo ({promedio_inter:.1f}s) "
                    "posible desatencion / mente en la nube"
                )
            else:
                nivel = 0.9
                categoria = "alta"
                motivo = "patron de parpadeo en rango esperado"
        else:
            # Pocas muestras -> no sabemos mucho, asumimos atención media-alta
            nivel = 0.8
            categoria = "media"
            motivo = "pocas muestras de parpadeo, asumiendo atencion media"

        # (Futuro) Respuestas a estímulos de la pulsera...
        return AtencionConductor(nivel=nivel, categoria=categoria, motivo=motivo)

    def obtener_estadisticas(self) -> dict:
        return {
            "parpadeos_total": self._conteo_parpadeos_total,
            "microsuenos_total": self._conteo_microsuenos_total,
            "bostezos_total": self._conteo_bostezos_total,
            "cabeceos_total": self._conteo_cabeceos_total,
        }
