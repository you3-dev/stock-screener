"""Product-facing overlay helpers (integration design docs/06, layer C state).

Surfaces the current regime (risk-on/off + a recommended position-size factor)
for the Streamlit banner and the nightly batch.  Risk management, not a
selection signal (finance §1).
"""
from __future__ import annotations

import logging

import pandas as pd

from src.overlay import regime

logger = logging.getLogger(__name__)

# size factors: cut new-entry sizing in risk-off (docs/10 GO = simple GSPC<50dMA)
SIZE_RISK_ON = 1.0
SIZE_RISK_OFF = 0.5


def current_regime() -> dict:
    """Latest risk regime state.

    Returns dict: asof, risk_off (bool), corr (float|None), size_factor,
    label.  Falls back to risk-on if macro is unavailable.
    """
    try:
        macro = regime.fetch_macro()
        reg = regime.compute_regime(macro).dropna(subset=["risk_off"])
        if reg.empty:
            raise ValueError("empty regime")
        last = reg.iloc[-1]
        risk_off = bool(last["risk_off"])
        corr = float(last["corr"]) if pd.notna(last["corr"]) else None
        return {
            "asof": pd.Timestamp(last["date"]).date().isoformat(),
            "risk_off": risk_off,
            "corr": corr,
            "size_factor": SIZE_RISK_OFF if risk_off else SIZE_RISK_ON,
            "label": "リスクオフ" if risk_off else "リスクオン",
        }
    except Exception as e:  # noqa: BLE001 - never break the app on macro failure
        logger.warning("regime unavailable: %s: %s", type(e).__name__, e)
        return {"asof": None, "risk_off": False, "corr": None,
                "size_factor": SIZE_RISK_ON, "label": "不明(マクロ取得不可)"}
