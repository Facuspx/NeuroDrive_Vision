"""
Módulo de cálculo de medidas geométricas del rostro
(EAR, MAR y medidas simples de cabeceo) a partir de los
puntos faciales detectados con MediaPipe FaceMesh.

Este módulo NO decide todavía si hay parpadeo, bostezo o cabeceo.
Solo entrega medidas numéricas robustas que luego serán usadas
por los módulos de detección de eventos de somnolencia.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

import numpy as np

from .detector_rostro_mediapipe import DatosRostro

logger = logging.getLogger(__name__)


# ==============================
#   Excepciones
# ==============================

class ErrorMedidasRostro(Exception):
    """Error genérico en el cálculo de medidas del rostro."""
    pass


# ==============================
#   Índices de referencia FaceMesh
# ==============================

# Nota: estos índices corresponden al modelo MediaPipe FaceMesh de 468 puntos.
# Pueden ajustarse si se calibra con otras referencias.

INDICES_FACEMESH: Dict[str, Dict[str, List[int] | int]] = {
    "ojos": {
        # Ojo izquierdo: 6 puntos (EAR estilo dlib adaptado a FaceMesh)
        # p0-p3: horizontales; p1-p5 y p2-p4: verticales
        "izquierdo": [33, 160, 158, 133, 153, 144],
        # Ojo derecho: índices equivalentes en el lado derecho
        "derecho": [362, 385, 387, 263, 373, 380],
    },
    "boca": {
        # Comisuras y puntos centrales para apertura
        "comisura_izquierda": 78,
        "comisura_derecha": 308,
        "labio_superior": 13,
        "labio_inferior": 14,
    },
    "cabeza": {
        # Puntos para features simples de posición
        "nariz": 1,
        "menton": 152,
    },
}


# ==============================
#   Estructuras de datos
# ==============================

@dataclass
class MedidasOjos:
    ear_izquierdo: Optional[float] = None
    ear_derecho: Optional[float] = None
    ear_promedio: Optional[float] = None
    valido: bool = False
    mensaje_error: Optional[str] = None


@dataclass
class MedidasBoca:
    mar: Optional[float] = None
    apertura_vertical_pixeles: Optional[float] = None
    ancho_boca_pixeles: Optional[float] = None
    valido: bool = False
    mensaje_error: Optional[str] = None


@dataclass
class MedidasCabeza:
    altura_relativa_nariz: Optional[float] = None
    altura_relativa_menton: Optional[float] = None
    distancia_nariz_menton_pixeles: Optional[float] = None
    valido: bool = False
    mensaje_error: Optional[str] = None


@dataclass
class MedidasRostro:
    """
    Contenedor de todas las medidas geométricas calculadas para un frame.
    """
    medidas_ojos: MedidasOjos = field(default_factory=MedidasOjos)
    medidas_boca: MedidasBoca = field(default_factory=MedidasBoca)
    medidas_cabeza: MedidasCabeza = field(default_factory=MedidasCabeza)
    rostro_presente: bool = False
    razones_no_valido: List[str] = field(default_factory=list)


# ==============================
#   Funciones auxiliares
# ==============================

def _distancia_2d(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """Distancia euclidiana 2D entre dos puntos en píxeles."""
    return float(np.linalg.norm(np.array(p1, dtype=float) - np.array(p2, dtype=float)))


def _obtener_punto(puntos: List[Tuple[int, int]], indice: int) -> Tuple[int, int]:
    """Obtiene un punto (x, y) de la lista de puntos_pixeles, con verificación de rango."""
    if indice < 0 or indice >= len(puntos):
        raise ErrorMedidasRostro(
            f"Índice de punto fuera de rango: {indice} (len={len(puntos)})"
        )
    return puntos[indice]


# ==============================
#   Clase principal
# ==============================

class CalculadorMedidasRostro:
    """
    Calcula medidas geométricas (EAR, MAR, etc.) a partir de DatosRostro.

    Uso típico:
        calculador = CalculadorMedidasRostro()
        medidas = calculador.calcular_medidas(datos_rostro)

    Luego otro módulo se encargará de usar estas medidas para detectar
    parpadeos, bostezos y cabeceos de manera robusta.
    """

    def __init__(self, config_indices: Optional[Dict[str, Dict]] = None) -> None:
        """
        Parameters
        ----------
        config_indices : dict | None
            Permite sobre-escribir los índices por defecto de FaceMesh.
            Si es None, se usa INDICES_FACEMESH.
        """
        self._indices = config_indices if config_indices is not None else INDICES_FACEMESH

    # ---------- API principal ----------

    def calcular_medidas(self, datos_rostro: DatosRostro) -> MedidasRostro:
        """
        Calcula todas las medidas geométricas principales a partir de DatosRostro.

        Returns
        -------
        MedidasRostro
        """
        medidas = MedidasRostro()
        medidas.rostro_presente = datos_rostro.rostro_presente

        if not datos_rostro.rostro_presente:
            medidas.razones_no_valido.append("No se detectó rostro en el frame.")
            return medidas

        if datos_rostro.puntos_pixeles is None or datos_rostro.resolucion is None:
            medidas.razones_no_valido.append(
                "Datos de puntos_pixeles o resolución no disponibles."
            )
            return medidas

        puntos = datos_rostro.puntos_pixeles
        ancho, alto = datos_rostro.resolucion

        # Cálculo de medidas de ojos (EAR)
        try:
            medidas_ojos = self._calcular_medidas_ojos(puntos)
            medidas.medidas_ojos = medidas_ojos
        except ErrorMedidasRostro as e:
            logger.warning(f"No se pudieron calcular medidas de ojos: {e}")
            medidas.medidas_ojos.mensaje_error = str(e)

        # Cálculo de medidas de boca (MAR)
        try:
            medidas_boca = self._calcular_medidas_boca(puntos)
            medidas.medidas_boca = medidas_boca
        except ErrorMedidasRostro as e:
            logger.warning(f"No se pudieron calcular medidas de boca: {e}")
            medidas.medidas_boca.mensaje_error = str(e)

        # Cálculo de medidas de cabeza (features simples)
        try:
            medidas_cabeza = self._calcular_medidas_cabeza(puntos, alto)
            medidas.medidas_cabeza = medidas_cabeza
        except ErrorMedidasRostro as e:
            logger.warning(f"No se pudieron calcular medidas de cabeza: {e}")
            medidas.medidas_cabeza.mensaje_error = str(e)

        # Si al menos una de las tres áreas es válida, consideramos que el conjunto
        # de medidas es potencialmente útil (la decisión final la hará el módulo de eventos).
        if not (medidas.medidas_ojos.valido or medidas.medidas_boca.valido or medidas.medidas_cabeza.valido):
            medidas.razones_no_valido.append(
                "Ninguna de las medidas principales (ojos, boca, cabeza) se pudo calcular correctamente."
            )

        return medidas

    # ---------- Medidas de ojos (EAR) ----------

    def _calcular_medidas_ojos(self, puntos: List[Tuple[int, int]]) -> MedidasOjos:
        idx_ojos = self._indices["ojos"]
        indices_izq = idx_ojos["izquierdo"]  # type: ignore
        indices_der = idx_ojos["derecho"]    # type: ignore

        # Asegurar que tenemos 6 puntos por ojo
        if len(indices_izq) != 6 or len(indices_der) != 6:
            raise ErrorMedidasRostro("Los índices de ojos deben tener exactamente 6 puntos por ojo.")

        # Ojo izquierdo
        p0 = _obtener_punto(puntos, indices_izq[0])
        p1 = _obtener_punto(puntos, indices_izq[1])
        p2 = _obtener_punto(puntos, indices_izq[2])
        p3 = _obtener_punto(puntos, indices_izq[3])
        p4 = _obtener_punto(puntos, indices_izq[4])
        p5 = _obtener_punto(puntos, indices_izq[5])

        # Fórmula EAR clásica: (||p1-p5|| + ||p2-p4||) / (2 * ||p0-p3||)
        ear_izq_num = _distancia_2d(p1, p5) + _distancia_2d(p2, p4)
        ear_izq_den = 2.0 * _distancia_2d(p0, p3)
        if ear_izq_den <= 0:
            raise ErrorMedidasRostro("Distancia horizontal del ojo izquierdo es cero.")
        ear_izq = ear_izq_num / ear_izq_den

        # Ojo derecho
        p0 = _obtener_punto(puntos, indices_der[0])
        p1 = _obtener_punto(puntos, indices_der[1])
        p2 = _obtener_punto(puntos, indices_der[2])
        p3 = _obtener_punto(puntos, indices_der[3])
        p4 = _obtener_punto(puntos, indices_der[4])
        p5 = _obtener_punto(puntos, indices_der[5])

        ear_der_num = _distancia_2d(p1, p5) + _distancia_2d(p2, p4)
        ear_der_den = 2.0 * _distancia_2d(p0, p3)
        if ear_der_den <= 0:
            raise ErrorMedidasRostro("Distancia horizontal del ojo derecho es cero.")
        ear_der = ear_der_num / ear_der_den

        ear_promedio = (ear_izq + ear_der) / 2.0

        return MedidasOjos(
            ear_izquierdo=ear_izq,
            ear_derecho=ear_der,
            ear_promedio=ear_promedio,
            valido=True,
        )

    # ---------- Medidas de boca (MAR simplificado) ----------

    def _calcular_medidas_boca(self, puntos: List[Tuple[int, int]]) -> MedidasBoca:
        idx_boca = self._indices["boca"]

        idx_com_izq = idx_boca["comisura_izquierda"]  # type: ignore
        idx_com_der = idx_boca["comisura_derecha"]    # type: ignore
        idx_lab_sup = idx_boca["labio_superior"]      # type: ignore
        idx_lab_inf = idx_boca["labio_inferior"]      # type: ignore

        com_izq = _obtener_punto(puntos, idx_com_izq)
        com_der = _obtener_punto(puntos, idx_com_der)
        lab_sup = _obtener_punto(puntos, idx_lab_sup)
        lab_inf = _obtener_punto(puntos, idx_lab_inf)

        ancho_boca = _distancia_2d(com_izq, com_der)
        apertura_vertical = _distancia_2d(lab_sup, lab_inf)

        if ancho_boca <= 0:
            raise ErrorMedidasRostro("Ancho de boca es cero o negativo.")

        mar = apertura_vertical / ancho_boca

        return MedidasBoca(
            mar=mar,
            apertura_vertical_pixeles=apertura_vertical,
            ancho_boca_pixeles=ancho_boca,
            valido=True,
        )

    # ---------- Medidas simples de cabeza ----------

    def _calcular_medidas_cabeza(
        self,
        puntos: List[Tuple[int, int]],
        alto: int
    ) -> MedidasCabeza:
        idx_cabeza = self._indices["cabeza"]

        idx_nariz = idx_cabeza["nariz"]   # type: ignore
        idx_menton = idx_cabeza["menton"] # type: ignore

        nariz = _obtener_punto(puntos, idx_nariz)
        menton = _obtener_punto(puntos, idx_menton)

        distancia_nariz_menton = _distancia_2d(nariz, menton)

        if alto <= 0:
            raise ErrorMedidasRostro("Altura de imagen inválida.")

        # Alturas relativas normalizadas (0 = tope superior, 1 = borde inferior)
        altura_rel_nariz = float(nariz[1]) / float(alto)
        altura_rel_menton = float(menton[1]) / float(alto)

        return MedidasCabeza(
            altura_relativa_nariz=altura_rel_nariz,
            altura_relativa_menton=altura_rel_menton,
            distancia_nariz_menton_pixeles=distancia_nariz_menton,
            valido=True,
        )
