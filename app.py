# app.py — GFI Bunkering Optimizer (Single Vessel)
# ------------------------------------------------
# Updated to match the revised output spec:
#  - Regulatory Cost = Tier1 + Tier2 + Benefit (from INITIAL GFI, not optimized)
#  - Premium Fuel Cost = Premium × Other_Fuel_Increase_For_Opt_Cost_<year> (increase vs initial Others)
#  - Total Cost = Regulatory Cost + Premium Fuel Cost
#
# Inputs persisted; defaults set to your requested values.
# Units:
#   Mass: tons
#   LCV:  MJ/ton  (your MJ/g defaults have been converted ×1000)
#   WtW:  gCO2eq/MJ

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Constants (aligned to the original model)
# ──────────────────────────────────────────────────────────────────────────────
GFI2008 = 93.3  # gCO2eq/MJ
ZT_BASE = {2028: 4.0, 2029: 6.0, 2030: 8.0, 2031: 12.4, 2032: 16.8, 2033: 21.2, 2034: 25.6, 2035: 30.0}
ZT_DIRECT = {2028: 17.0, 2029: 19.0, 2030: 21.0, 2031: 25.4, 2032: 29.8, 2033: 34.2, 2034: 38.6, 2035: 43.0}
YEARS = list(range(2028, 2036))

# Cost rates (USD per tCO2eq)
TIER1_COST = 100.0
TIER2_COST = 380.0
BENEFIT_RATE = 380.0  # negative mass → negative $, credit

# Persistence
DEFAULTS_PATH = ".gfi_bunkering_defaults.json"

CF_LABELS = {
    "HFO": "HFO (Cf: 3.114)",
    "LFO": "LFO (Cf: 3.151)",
    "MDO": "MDO/MGO (Cf: 3.206)",
    "OTH": "Others (Cf: —)",
}

# Targets as absolute g/MJ
GFI_BASE = {yr: (1 - ZT_BASE[yr] / 100.0) * GFI2008 for yr in YEARS}
GFI_DIRECT = {yr: (1 - ZT_DIRECT[yr] / 100.0) * GFI2008 for yr in YEARS}

# ──────────────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class FuelInputs:
    HFO_t: float
    LFO_t: float
    MDO_t: float
    OTH_t: float
    WtW_HFO: float
    WtW_LFO: float
    WtW_MDO: float
    WtW_OTH: float
    LCV_HFO: float  # MJ/ton
    LCV_LFO: float
    LCV_MDO: float
    LCV_OTH: float
    PREMIUM: float  # USD/ton

    def total_MJ(self) -> float:
        return (
            self.HFO_t * self.LCV_HFO
            + self.LFO_t * self.LCV_LFO
            + self.MDO_t * self.LCV_MDO
            + self.OTH_t * self.LCV_OTH
        )

    def gfi(self) -> float:
        mj = self.total_MJ()
        if mj <= 0:
            return 0.0
        num_g = (
            self.HFO_t * self.LCV_HFO * self.WtW_HFO
            + self.LFO_t * self.LCV_LFO * self.WtW_LFO
            + self.MDO_t * self.LCV_MDO * self.WtW_MDO
            + self.OTH_t * self.LCV_OTH * self.WtW_OTH
        )
        return num_g / mj  # g/MJ


# ──────────────────────────────────────────────────────────────────────────────
# Helpers for persisted defaults
# ──────────────────────────────────────────────────────────────────────────────
def load_defaults() -> Dict:
    try:
        with open(DEFAULTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_defaults(dct: Dict) -> None:
    try:
        with open(DEFAULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(dct, f, indent=2)
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Core calculations
# ──────────────────────────────────────────────────────────────────────────────
def deficit_surplus_tCO2eq(gfi_g_per_MJ: float, total_MJ: float, year: int) -> float:
    """Return GFI_Deficit_Surplus_<year> in tCO2eq (positive=deficit, negative=surplus)."""
    base = GFI_BASE[year]
    direct = GFI_DIRECT[year]

    if gfi_g_per_MJ > base:
        tier2 = gfi_g_per_MJ - base
        tier1 = base - direct
        g_g = (tier1 + tier2) * total_MJ
    elif gfi_g_per_MJ >= direct:
        g_g = (gfi_g_per_MJ - direct) * total_MJ
    else:
        g_g = (gfi_g_per_MJ - direct) * total_MJ  # negative

    return g_g / 1e6  # grams → tonnes


def tier_costs_usd(gfi_g_per_MJ: float, total_MJ: float, year: int) -> Tuple[float, float, float]:
    """Return (Tier1 USD, Tier2 USD, Benefit USD) from INITIAL (pre-optimization) GFI."""
    base = GFI_BASE[year]
    direct = GFI_DIRECT[year]

    if gfi_g_per_MJ > base:
        tier1_mt = (base - direct) * total_MJ / 1e6
        tier2_mt = (gfi_g_per_MJ - base) * total_MJ / 1e6
        benefit_mt = 0.0
    elif gfi_g_per_MJ >= direct:
        tier1_mt = (gfi_g_per_MJ - direct) * total_MJ / 1e6
        tier2_mt = 0.0
        benefit_mt = 0.0
    else:
        tier1_mt = 0.0
        tier2_mt = 0.0
        benefit_mt = (gfi_g_per_MJ - direct) * total_MJ / 1e6  # negative

    return tier1_mt * TIER1_COST, tier2_mt * TIER2_COST, benefit_mt * BENEFIT_RATE


def optimize_energy_neutral(
    fi: FuelInputs,
    year: int,
    coarse_steps: int = 50,
    fine_window: float = 0.04,
    fine_step: float = 0.005,
) -> Tuple[float, float, float, float, float]:
    """Energy-neutral HFO→Others swap, per year.
    Returns: (hfo_red_t, oth_inc_t, gfi_new, reg_cost_usd, premium_cost_usd)

    - Reduce HFO by a fraction f of its initial mass.
    - Increase Others by ΔOTH = (ΔHFO * LCV_HFO) / LCV_OTH (keeps total MJ constant).
    - Compute GFI_new and Tier costs for that year.
    - Regulatory cost here is for the *post-swap* GFI; we DO NOT use it in 'Regulatory Cost' output.
    - Premium cost = PREMIUM × max(ΔOTH, 0) with ΔOTH measured vs initial Others.
    """
    if fi.HFO_t <= 0 or fi.LCV_OTH <= 0:
        return 0.0, 0.0, fi.gfi(), 0.0, 0.0

    total_MJ0 = fi.total_MJ()

    def eval_fraction(f: float) -> Tuple[float, float, float, float, float]:
        hfo_new = fi.HFO_t * (1.0 - f)
        d_hfo = fi.HFO_t - hfo_new
        oth_new = fi.OTH_t + (d_hfo * fi.LCV_HFO) / fi.LCV_OTH

        total_MJ = (
            hfo_new * fi.LCV_HFO
            + fi.LFO_t * fi.LCV_LFO
            + fi.MDO_t * fi.LCV_MDO
            + oth_new * fi.LCV_OTH
        )
        if total_MJ <= 0:
            return np.inf, 0.0, 0.0, 0.0, 0.0

        # New GFI
        num_g = (
            hfo_new * fi.LCV_HFO * fi.WtW_HFO
            + fi.LFO_t * fi.LCV_LFO * fi.WtW_LFO
            + fi.MDO_t * fi.LCV_MDO * fi.WtW_MDO
            + oth_new * fi.LCV_OTH * fi.WtW_OTH
        )
        gfi_new = num_g / total_MJ

        # Post-swap tier costs (used only for internal optimization objective)
        t1_usd, t2_usd, ben_usd = tier_costs_usd(gfi_new, total_MJ, year)
        reg_cost_usd = t1_usd + t2_usd + ben_usd
        premium_cost_usd = max(oth_new - fi.OTH_t, 0.0) * fi.PREMIUM
        total_cost = reg_cost_usd + premium_cost_usd
        return total_cost, gfi_new, reg_cost_usd, premium_cost_usd, d_hfo

    # Coarse sweep
    grid = np.linspace(0.0, 1.0, coarse_steps + 1)
    best = None
    for f in grid:
        tot, gfi_new, reg, prem, d_hfo = eval_fraction(f)
        if best is None or tot < best[0]:
            best = (tot, f, gfi_new, reg, prem, d_hfo)

    _, f_best, gfi_best, reg_best, prem_best, d_hfo_best = best

    # Fine sweep near best
    lo = max(0.0, f_best - fine_window)
    hi = min(1.0, f_best + fine_window)
    f = lo
    while f <= hi + 1e-12:
        tot, gfi_new, reg, prem, d_hfo = eval_fraction(f)
        if tot < reg_best + prem_best - 1e-12:
            f_best, gfi_best, reg_best, prem_best, d_hfo_best = f, gfi_new, reg, prem, d_hfo
        f += fine_step

    hfo_red = d_hfo_best
    oth_inc = (hfo_red * fi.LCV_HFO) / fi.LCV_OTH if fi.LCV_OTH > 0 else 0.0
    return hfo_red, oth_inc, gfi_best, reg_best, prem_best


# ──────────────────────────────────────────────────────────────────────────────
# UI — Streamlit
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="GFI Bunkering Optimizer — Single Vessel", layout="wide")
st.title("GFI Bunkering Optimizer — Single Vessel")

with st.expander("Methodology & Units", expanded=False):
    st.markdown(
        """
**Formulas (as in original):**

- **GFI** \[gCO₂e/MJ] = \( \sum m_i·LCV_i·WtW_i \) / \( \sum m_i·LCV_i \)  
- **Deficit/Surplus** \[tCO₂e] for year *y*:
  - If GFI > Base_y: (GFI−Base_y + Base_y−Direct_y)·TotalMJ / 10⁶
  - If Direct_y ≤ GFI ≤ Base_y: (GFI−Direct_y)·TotalMJ / 10⁶
  - If GFI < Direct_y: (GFI−Direct_y)·TotalMJ / 10⁶ (negative = surplus)
- **Tier rates**: Tier-1 = 100, Tier-2 = 380, Benefit = 380 (USD per tCO₂e)
- **Optimization (per year)**: energy-neutral HFO→Others swap; objective = Tier costs (post-swap) + Premium×ΔOthers.  
  *Outputs “Regulatory Cost”, “Premium Fuel Cost”, “Total Cost” use the initial (pre-optimization) Tier costs and the ΔOthers vs initial.*
        """
    )

# Load persisted defaults
states = load_defaults()

# Sidebar inputs (defaults set to your requested values; editable)
st.sidebar.header("Inputs (persisted)")
colA, colB = st.sidebar.columns(2)
HFO_t = colA.number_input(f"{CF_LABELS['HFO']} — Tons", min_value=0.0, value=float(states.get("HFO_t", 100.0)), step=0.1)
LFO_t = colB.number_input(f"{CF_LABELS['LFO']} — Tons", min_value=0.0, value=float(states.get("LFO_t", 0.0)), step=0.1)
MDO_t = colA.number_input(f"{CF_LABELS['MDO']} — Tons", min_value=0.0, value=float(states.get("MDO_t", 0.0)), step=0.1)
OTH_t = colB.number_input(f"{CF_LABELS['OTH']} — Tons", min_value=0.0, value=float(states.get("OTH_t", 0.0)), step=0.1)

st.sidebar.markdown("---")
colC, colD = st.sidebar.columns(2)
WtW_HFO = colC.number_input("WtW HFO [gCO₂e/MJ]", min_value=0.0, value=float(states.get("WtW_HFO", 92.784)), step=0.001, format="%.3f")
WtW_LFO = colD.number_input("WtW LFO [gCO₂e/MJ]", min_value=0.0, value=float(states.get("WtW_LFO", 91.251)), step=0.001, format="%.3f")
WtW_MDO = colC.number_input("WtW MDO/MGO [gCO₂e/MJ]", min_value=0.0, value=float(states.get("WtW_MDO", 93.932)), step=0.001, format="%.3f")
WtW_OTH = colD.number_input("WtW Others [gCO₂e/MJ]", min_value=0.0, value=float(states.get("WtW_OTH", 70.366)), step=0.001, format="%.3f")

st.sidebar.markdown("---")
colE, colF = st.sidebar.columns(2)
# Convert your MJ/g defaults to MJ/ton (×1000) for the engine
LCV_HFO = colE.number_input("LCV HFO [MJ/ton]", min_value=0.0, value=float(states.get("LCV_HFO", 40200.0)), step=100.0)
LCV_LFO = colF.number_input("LCV LFO [MJ/ton]", min_value=0.0, value=float(states.get("LCV_LFO", 41000.0)), step=100.0)
LCV_MDO = colE.number_input("LCV MDO/MGO [MJ/ton]", min_value=0.0, value=float(states.get("LCV_MDO", 42700.0)), step=100.0)
LCV_OTH = colF.number_input("LCV Others [MJ/ton]", min_value=0.0, value=float(states.get("LCV_OTH", 37000.0)), step=100.0)

st.sidebar.markdown("---")
PREMIUM = st.sidebar.number_input("Premium [USD/ton] (Others − HFO)", min_value=0.0, value=float(states.get("PREMIUM", 305.0)), step=5.0)

if st.sidebar.button("Save as defaults", use_container_width=True):
    save_defaults({
        "HFO_t": HFO_t, "LFO_t": LFO_t, "MDO_t": MDO_t, "OTH_t": OTH_t,
        "WtW_HFO": WtW_HFO, "WtW_LFO": WtW_LFO, "WtW_MDO": WtW_MDO, "WtW_OTH": WtW_OTH,
        "LCV_HFO": LCV_HFO, "LCV_LFO": LCV_LFO, "LCV_MDO": LCV_MDO, "LCV_OTH": LCV_OTH,
        "PREMIUM": PREMIUM,
    })
    st.sidebar.success("Defaults saved.")

# Build inputs object
fi = FuelInputs(
    HFO_t=HFO_t, LFO_t=LFO_t, MDO_t=MDO_t, OTH_t=OTH_t,
    WtW_HFO=WtW_HFO, WtW_LFO=WtW_LFO, WtW_MDO=WtW_MDO, WtW_OTH=WtW_OTH,
    LCV_HFO=LCV_HFO, LCV_LFO=LCV_LFO, LCV_MDO=LCV_MDO, LCV_OTH=LCV_OTH,
    PREMIUM=PREMIUM,
)

# ──────────────────────────────────────────────────────────────────────────────
# Base metrics
# ──────────────────────────────────────────────────────────────────────────────
TOTAL_MJ = fi.total_MJ()
GFI = fi.gfi()

k1, k2 = st.columns(2)
k1.metric("GFI (gCO₂e/MJ)", f"{GFI:.3f}")
k2.metric("Total energy (MJ)", f"{TOTAL_MJ:,.0f}")

# 1) GFI plot vs Base/Direct/Baseline
fig = go.Figure()
fig.add_trace(go.Scatter(x=YEARS, y=[GFI]*len(YEARS), mode="lines", name="GFI attained", line=dict(width=3)))
fig.add_trace(go.Scatter(x=YEARS, y=[GFI_BASE[y] for y in YEARS], mode="lines", name="Base target", line=dict(dash="dash")))
fig.add_trace(go.Scatter(x=YEARS, y=[GFI_DIRECT[y] for y in YEARS], mode="lines", name="Direct target", line=dict(dash="dot")))
fig.add_trace(go.Scatter(x=YEARS, y=[GFI2008]*len(YEARS), mode="lines", name="Baseline 2008", line=dict(color="black", dash="longdash")))
fig.update_layout(height=360, margin=dict(l=10,r=10,t=40,b=10), yaxis_title="gCO₂e/MJ", xaxis_title="Year")
st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Per-year outputs
# ──────────────────────────────────────────────────────────────────────────────
rows: List[Dict] = []
for yr in YEARS:
    # Initial (pre-optimization) quantities drive deficits & tier costs
    deficit_t = deficit_surplus_tCO2eq(GFI, TOTAL_MJ, yr)
    t1_usd, t2_usd, ben_usd = tier_costs_usd(GFI, TOTAL_MJ, yr)
    regulatory_cost_usd = t1_usd + t2_usd + ben_usd  # INITIAL (as required)

    # Optimization (energy-neutral HFO→Others), per year
    hfo_red_t, oth_inc_t, gfi_new, reg_cost_post, prem_cost_post = optimize_energy_neutral(fi, yr)

    # Premium Fuel Cost uses ΔOthers vs initial
    premium_fuel_cost_usd = PREMIUM * oth_inc_t

    # Total Cost per requirement
    total_cost_usd = regulatory_cost_usd + premium_fuel_cost_usd

    rows.append({
        "Year": yr,
        "GFI (g/MJ)": round(GFI, 6),
        # 2
        "GFI_Deficit_Surplus_tCO2eq": deficit_t,
        # 3–5 (initial)
        "GFI_Tier_1_Cost_USD": t1_usd,
        "GFI_Tier_2_Cost_USD": t2_usd,
        "GFI_Benefit_USD": ben_usd,
        # 6–7 (optimization deltas)
        "HFO_Reduction_For_Opt_Cost_t": hfo_red_t,
        "Other_Fuel_Increase_For_Opt_Cost_t": oth_inc_t,
        # 8–10 (as specified)
        "Regulatory_Cost_USD": regulatory_cost_usd,
        "Premium_Fuel_Cost_USD": premium_fuel_cost_usd,
        "Total_Cost_USD": total_cost_usd,
        # (optional visibility) post-swap values not used in required outputs:
        "GFI_Optimized (info)": gfi_new,
        "Reg_Cost_PostSwap (info)": reg_cost_post,
    })

res_df = pd.DataFrame(rows)

st.subheader("Per-Year Results (2028–2035)")
st.dataframe(
    res_df[
        [
            "Year",
            "GFI (g/MJ)",
            "GFI_Deficit_Surplus_tCO2eq",
            "GFI_Tier_1_Cost_USD",
            "GFI_Tier_2_Cost_USD",
            "GFI_Benefit_USD",
            "HFO_Reduction_For_Opt_Cost_t",
            "Other_Fuel_Increase_For_Opt_Cost_t",
            "Regulatory_Cost_USD",
            "Premium_Fuel_Cost_USD",
            "Total_Cost_USD",
        ]
    ].style.format({
        "GFI (g/MJ)": "{:.3f}",
        "GFI_Deficit_Surplus_tCO2eq": "{:.3f}",
        "GFI_Tier_1_Cost_USD": "{:,.0f}",
        "GFI_Tier_2_Cost_USD": "{:,.0f}",
        "GFI_Benefit_USD": "{:,.0f}",
        "HFO_Reduction_For_Opt_Cost_t": "{:.3f}",
        "Other_Fuel_Increase_For_Opt_Cost_t": "{:.3f}",
        "Regulatory_Cost_USD": "{:,.0f}",
        "Premium_Fuel_Cost_USD": "{:,.0f}",
        "Total_Cost_USD": "{:,.0f}",
    }),
    use_container_width=True,
    height=400
)

# ──────────────────────────────────────────────────────────────────────────────
# Required bar charts: 2, 8, 9, 10
# ──────────────────────────────────────────────────────────────────────────────
def bar_chart(title: str, ycol: str):
    figb = px.bar(res_df, x="Year", y=ycol, title=title)
    figb.update_layout(height=320, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(figb, use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    bar_chart("GFI Deficit / Surplus [tCO₂e] (Item 2)", "GFI_Deficit_Surplus_tCO2eq")
with c2:
    bar_chart("Regulatory Cost [USD] (Item 8)", "Regulatory_Cost_USD")

c3, c4 = st.columns(2)
with c3:
    bar_chart("Premium Fuel Cost [USD] (Item 9)", "Premium_Fuel_Cost_USD")
with c4:
    bar_chart("Total Cost [USD] (Item 10)", "Total_Cost_USD")

# Optional: show Tier breakdown bars for transparency
with st.expander("Details: Tier breakdown (initial, pre-optimization)"):
    c5, c6, c7 = st.columns(3)
    with c5:
        bar_chart("Tier 1 Cost [USD]", "GFI_Tier_1_Cost_USD")
    with c6:
        bar_chart("Tier 2 Cost [USD]", "GFI_Tier_2_Cost_USD")
    with c7:
        bar_chart("Benefit [USD] (negative = credit)", "GFI_Benefit_USD")

# ──────────────────────────────────────────────────────────────────────────────
# Excel export
# ──────────────────────────────────────────────────────────────────────────────
buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as xw:
    pd.DataFrame({
        "Parameter": ["HFO_t","LFO_t","MDO_t","OTH_t","WtW_HFO","WtW_LFO","WtW_MDO","WtW_OTH","LCV_HFO","LCV_LFO","LCV_MDO","LCV_OTH","PREMIUM"],
        "Value": [HFO_t,LFO_t,MDO_t,OTH_t,WtW_HFO,WtW_LFO,WtW_MDO,WtW_OTH,LCV_HFO,LCV_LFO,LCV_MDO,LCV_OTH,PREMIUM],
        "Units": ["t","t","t","t","g/MJ","g/MJ","g/MJ","g/MJ","MJ/t","MJ/t","MJ/t","MJ/t","USD/t"],
    }).to_excel(xw, sheet_name="Inputs", index=False)
    res_df.to_excel(xw, sheet_name="Results_2028_2035", index=False)

st.download_button(
    label="Download results (Excel)",
    data=buf.getvalue(),
    file_name="GFI_Bunkering_Optimizer_SingleVessel.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.caption("© 2025 — Single-vessel GFI optimizer. Regulatory & Premium costs per your updated specification.")
