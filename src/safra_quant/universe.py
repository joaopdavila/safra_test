"""IBRX 50 universe (B3 composition, September 2022)."""

from __future__ import annotations

IBRX50_SEP_2022: tuple[str, ...] = (
    "ABEV3.SA",
    "AMER3.SA",
    "ASAI3.SA",
    "AZUL4.SA",
    "B3SA3.SA",
    "BBAS3.SA",
    "BBDC4.SA",
    "BBSE3.SA",
    "BPAC11.SA",
    "BRAP4.SA",
    "BRFS3.SA",
    "BRKM5.SA",
    "BRML3.SA",
    "CASH3.SA",
    "CCRO3.SA",
    "CSAN3.SA",
    "CSNA3.SA",
    "CVCB3.SA",
    "ELET3.SA",
    "ELET6.SA",
    "EMBR3.SA",
    "EQTL3.SA",
    "GGBR4.SA",
    "HAPV3.SA",
    "ITSA4.SA",
    "ITUB4.SA",
    "JBSS3.SA",
    "KLBN11.SA",
    "LREN3.SA",
    "LWSA3.SA",
    "MGLU3.SA",
    "MRFG3.SA",
    "MULT3.SA",
    "NTCO3.SA",
    "PETR3.SA",
    "PETR4.SA",
    "PETZ3.SA",
    "PRIO3.SA",
    "RADL3.SA",
    "RAIL3.SA",
    "RDOR3.SA",
    "RENT3.SA",
    "RRRP3.SA",
    "SUZB3.SA",
    "TOTS3.SA",
    "USIM5.SA",
    "VALE3.SA",
    "VBBR3.SA",
    "VIIA3.SA",
    "WEGE3.SA",
)

IBOVESPA_TICKER: str = "^BVSP"


def get_ibrx50() -> list[str]:
    return list(IBRX50_SEP_2022)
