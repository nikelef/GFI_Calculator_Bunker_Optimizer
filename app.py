# app.py — GFI Bunkering Optimizer (Single Vessel)
# ------------------------------------------------
# A self-contained Streamlit web app that turns your original Excel-based logic
# into a user-friendly tool for a *single vessel*.
#
# Key features (unchanged)
# • Inputs (persisted between runs):
#   - Fuel quantities [tons] for HFO (Cf:3.114), LFO (Cf:3.151), MDO/MGO (Cf:3.206), Others (Cf:—)
#   - WtW intensities [gCO2eq/MJ] for each fuel
#   - LCVs [MJ/ton] for each fuel (NOTE: original logic used ton-based mass)
#   - Premium [USD/ton] = price difference (Others − Selected Fuel)
# • Outputs for 2028–2035 (unchanged):
#   - GFI (gCO2eq/MJ) + plot vs. Base/Direct targets + baseline
#   - GFI_Deficit_Surplus_year [tCO2eq]
#   - GFI_Tier_1_Cost_year [USD]
#   - GFI_Tier_2_Cost_year [USD]
#   - GFI_Benefit_year [USD]
#   - <SelectedFuel>_Reduction_For_Opt_Cost_year [tons]
#   - Other_Fuel_Increase_For_Opt_Cost_year [tons]
#   - Regulatory Cost = Tier1 + Tier2 + Benefit  (INITIAL values; not post-optimization)
#   - Premium Fuel Cost = Premium × Others (INITIAL Others; not post-optimization)
#   - Total Cost = Regulatory Cost + Premium Fuel Cost
# • Energy-neutral optimization per YEAR: reduce the **selected fossil fuel**
#   (HFO or LFO or MDO/MGO), increase Others so total MJ stays constant;
#   objective = min(RegulatoryCost + PremiumCost).  (Used only to report deltas,
#   costs shown remain the INITIAL ones as in the original code.)
#
# How to run
#   1) pip install streamlit plotly pandas openpyxl
#   2) streamlit run app.py
#
# Notes
# - All constants (GFI2008, ZT_Base, ZT_Direct, Tier rates) match your original code.
# - Units are enforced to be consistent with the original math:
#     * LCV: MJ/ton
#     * WtW: gCO2eq/MJ
#     * Mass: tons
#   This way, (g/MJ)×(MJ) → gCO2eq; we convert to tCO2eq by ÷1e6.

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
# Constants (kept as per original code)
# ──────────────────────────────────────────────────────────────────────────────
GFI2008 = 93.3  # gCO2eq/MJ (baseline intensity)
ZT_BASE = {
    2028: 4.0,
    2029: 6.0,
    2030: 8.0,
    2031: 12.4,
    2032: 16.8,
    2033: 21.2,
    2034: 25.6,
    2035: 30.0,
}
ZT_DIRECT = {
    2028: 17.0,
    2029: 19.0,
    2030: 21.0,
    2031: 25.4,
    2032: 29.8,
    2033: 34.2,
    2034: 38.6,
    2035: 43.0,
}
YEARS = list(range(2028, 2036))

# Cost rates (USD per tCO2eq)
TIER1_COST = 100.0
TIER2_COST = 380.0
BENEFIT_RATE = 380.0  # negative mass → negative $ (benefit/credit)

# Default persistence file
DEFAULTS_PATH = ".gfi_bunkering_defaults.json"

# Labels (Cf just for UI display)
CF_LABELS = {
    "HFO": "HFO (Cf: 3.114)",
    "LFO": "LFO (Cf: 3.151)",
    "MDO": "MDO/MGO (Cf: 3.206)",
    "OTH": "Others (Cf: —)",
}

# ──────────────────────────────────────────────────────────────────────────────
# Data models & utils
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class FuelInputs:
    # Masses in tons
    HFO_t: float
    LFO_t: float
    MDO_t: float
    OTH_t: float

    # WtW in gCO2eq/MJ
    WtW_HFO: float
    WtW_LFO: float
    WtW_MDO: float
    WtW_OTH: float

    # LCV in MJ/ton
    LCV_HFO: float
    LCV_LFO: float
    LCV_MDO: float
    LCV_OTH: float

    # Premium USD/ton (Others − Selected Fuel)
    PREMIUM: float

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
        return num_g / mj  # gCO2eq/MJ


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


# Targets
GFI_BASE = {yr: (1 - ZT_BASE[yr] / 100.0) * GFI2008 for yr in YEARS}
GFI_DIRECT = {yr: (1 - ZT_DIRECT[yr] / 100.0) * GFI2008 for yr in YEARS}

# ──────────────────────────────────────────────────────────────────────────────
# Core calculations
# ──────────────────────────────────────────────────────────────────────────────
def deficit_surplus_tCO2eq(gfi_g_per_MJ: float, total_MJ: float, year: int) -> float:
    """Return GFI_Deficit_Surplus_year in tCO2eq.
    Sign convention:
      • above Direct → positive deficit (cost exposure)
      • below Direct → negative (surplus/credit)
    """
    base = GFI_BASE[year]
    direct = GFI_DIRECT[year]

    if gfi_g_per_MJ > base:
        tier2 = gfi_g_per_MJ - base
        tier1 = base - direct
        g_g = (tier1 + tier2) * total_MJ  # grams
    elif gfi_g_per_MJ >= direct:
        g_g = (gfi_g_per_MJ - direct) * total_MJ
    else:
        g_g = (gfi_g_per_MJ - direct) * total_MJ  # negative

    return g_g / 1e6  # → tonnes CO2eq


def tier_costs_usd(gfi_g_per_MJ: float, total_MJ: float, year: int) -> Tuple[float, float, float]:
    """Return (Tier1 USD, Tier2 USD, Benefit USD) for the year.
    Benefit is negative if below Direct.
    """
    base = GFI_BASE[year]
    direct = GFI_DIRECT[year]

    # masses in tonnes
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

    t1_usd = tier1_mt * TIER1_COST
    t2_usd = tier2_mt * TIER2_COST
    ben_usd = benefit_mt * BENEFIT_RATE  # negative → credit
    return t1_usd, t2_usd, ben_usd


def optimize_energy_neutral(
    fi: FuelInputs,
    year: int,
    reduce_fuel: str = "HFO",  # "HFO" | "LFO" | "MDO/MGO"
    coarse_steps: int = 200,
    fine_window: float = 0.02,
    fine_step: float = 0.001,
) -> Tuple[float, float, float, float, float]:
    """Per-year optimization.

    Reduce the selected fossil fuel (HFO or LFO or MDO/MGO), increase Others to keep MJ constant.

    Returns: (sel_red_t, oth_inc_t, gfi_new, reg_cost_usd, premium_cost_usd)
    """
    # Map selection to masses, WtW, LCV
    if reduce_fuel == "HFO":
        sel_mass0 = fi.HFO_t
        sel_lcv = fi.LCV_HFO
        sel_wtw = fi.WtW_HFO
    elif reduce_fuel == "LFO":
        sel_mass0 = fi.LFO_t
        sel_lcv = fi.LCV_LFO
        sel_wtw = fi.WtW_LFO
    else:  # "MDO/MGO"
        sel_mass0 = fi.MDO_t
        sel_lcv = fi.LCV_MDO
        sel_wtw = fi.WtW_MDO

    if sel_mass0 <= 0 or fi.LCV_OTH <= 0:
        return 0.0, 0.0, fi.gfi(), 0.0, 0.0

    # Baseline for reference
    total_MJ0 = fi.total_MJ()

    def total_cost_for_fraction(f: float) -> Tuple[float, float, float, float, float]:
        # New masses after reducing selected fuel by fraction f and increasing Others to keep energy constant
        sel_new = sel_mass0 * (1.0 - f)
        d_sel = sel_mass0 - sel_new
        oth_new = fi.OTH_t + (d_sel * sel_lcv) / fi.LCV_OTH

        # Compose per-fuel new masses
        hfo_new = fi.HFO_t
        lfo_new = fi.LFO_t
        mdo_new = fi.MDO_t
        if reduce_fuel == "HFO":
            hfo_new = sel_new
        elif reduce_fuel == "LFO":
            lfo_new = sel_new
        else:
            mdo_new = sel_new

        total_MJ = (
            hfo_new * fi.LCV_HFO
            + lfo_new * fi.LCV_LFO
            + mdo_new * fi.LCV_MDO
            + oth_new * fi.LCV_OTH
        )
        if total_MJ <= 0:
            return np.inf, fi.gfi(), 0.0, 0.0, 0.0

        num_g = (
            hfo_new * fi.LCV_HFO * fi.WtW_HFO
            + lfo_new * fi.LCV_LFO * fi.WtW_LFO
            + mdo_new * fi.LCV_MDO * fi.WtW_MDO
            + oth_new * fi.LCV_OTH * fi.WtW_OTH
        )
        gfi_new = num_g / total_MJ

        # Regulatory cost at this GFI for this year
        t1, t2, ben = tier_costs_usd(gfi_new, total_MJ, year)
        reg_cost = t1 + t2 + ben

        # Premium cost only for the added Others (relative to initial Others)
        premium_cost = max(oth_new - fi.OTH_t, 0.0) * fi.PREMIUM

        total_cost = reg_cost + premium_cost
        return total_cost, gfi_new, reg_cost, premium_cost, d_sel

    # Coarse sweep over f ∈ {0, 1/coarse_steps, ..., 1}
    grid = np.linspace(0.0, 1.0, coarse_steps + 1)
    best = None
    for f in grid:
        tot, gfi_new, reg, prem, d_sel = total_cost_for_fraction(f)
        if best is None or tot < best[0]:
            best = (tot, f, gfi_new, reg, prem, d_sel)

    _, f_best, gfi_best, reg_best, prem_best, d_sel_best = best

    # Fine local search around f_best
    lo = max(0.0, f_best - fine_window)
    hi = min(1.0, f_best + fine_window)
    f = lo
    while f <= hi + 1e-12:
        tot, gfi_new, reg, prem, d_sel = total_cost_for_fraction(f)
        if tot < (reg_best + prem_best) - 1e-12:
            f_best, gfi_best, reg_best, prem_best, d_sel_best = f, gfi_new, reg, prem, d_sel
        f += fine_step

    # Final masses deltas
    sel_red = d_sel_best
    oth_inc = (sel_red * sel_lcv) / fi.LCV_OTH if fi.LCV_OTH > 0 else 0.0
    return sel_red, oth_inc, gfi_best, reg_best, prem_best


# ──────────────────────────────────────────────────────────────────────────────
# UI — Streamlit
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="GFI Bunkering Optimizer — Single Vessel", layout="wide")
st.title("GFI Bunkering Optimizer — TMS NE")

with st.expander("Methodology & Units", expanded=False):
    st.markdown(
        """
        **Formulas (as in your original code):**
        
        - **GFI** \[gCO₂e/MJ] = \(\sum_i m_i·LCV_i·WtW_i\) / \(\sum_i m_i·LCV_i\)
        - **Deficit/Surplus** \[tCO₂e] for year *y*:
            - If GFI > Base_y: (GFI−Base_y + Base_y−Direct_y)·TotalMJ / 10⁶
            - If Direct_y ≤ GFI ≤ Base_y: (GFI−Direct_y)·TotalMJ / 10⁶
            - If GFI < Direct_y: (GFI−Direct_y)·TotalMJ / 10⁶ (negative surplus)
        - **Tier costs** \[USD]: Tier-1 = 100, Tier-2 = 380, Benefit = 380 × (negative mass)
        - **Optimization (per year):** reduce **selected fuel (HFO/LFO/MDO-MGO)** by Δ (tons)
          and increase **Others** by Δ·LCV_selected/LCV_OTH (energy-neutral). Objective:
          minimize *(Tier1 + Tier2 + Benefit + Premium·max(ΔOTH,0))*.
        
        **Units**: Mass in tons; LCV in MJ/ton; WtW in gCO₂e/MJ.
        """
    )

# Load persisted defaults (first)
states = load_defaults()

# Sidebar inputs — grouped
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
LCV_HFO = colE.number_input("LCV HFO [MJ/ton]", min_value=0.0, value=float(states.get("LCV_HFO", 40200.0)), step=100.0)
LCV_LFO = colF.number_input("LCV LFO [MJ/ton]", min_value=0.0, value=float(states.get("LCV_LFO", 41000.0)), step=100.0)
LCV_MDO = colE.number_input("LCV MDO/MGO [MJ/ton]", min_value=0.0, value=float(states.get("LCV_MDO", 42700.0)), step=100.0)
LCV_OTH = colF.number_input("LCV Others [MJ/ton]", min_value=0.0, value=float(states.get("LCV_OTH", 37000.0)), step=100.0)

st.sidebar.markdown("---")
# Select which fossil fuel to reduce
reduce_choice = st.sidebar.selectbox(
    "Fuel to reduce (for optimization)",
    options=["HFO", "LFO", "MDO/MGO"],
    index=int(states.get("reduce_idx", 0))
)

# Premium now refers to (Others − Selected Fuel)
PREMIUM = st.sidebar.number_input(
    f"Premium [USD/ton] (Others − {reduce_choice})",
    min_value=0.0,
    value=float(states.get("PREMIUM", 305.0)),
    step=10.0
)

if st.sidebar.button("Save as defaults", use_container_width=True):
    new_states = {
        "HFO_t": HFO_t, "LFO_t": LFO_t, "MDO_t": MDO_t, "OTH_t": OTH_t,
        "WtW_HFO": WtW_HFO, "WtW_LFO": WtW_LFO, "WtW_MDO": WtW_MDO, "WtW_OTH": WtW_OTH,
        "LCV_HFO": LCV_HFO, "LCV_LFO": LCV_LFO, "LCV_MDO": LCV_MDO, "LCV_OTH": LCV_OTH,
        "PREMIUM": PREMIUM, "reduce_idx": ["HFO","LFO","MDO/MGO"].index(reduce_choice)
    }
    save_defaults(new_states)
    st.sidebar.success("Defaults saved.")

# Build FuelInputs
fi = FuelInputs(
    HFO_t=HFO_t, LFO_t=LFO_t, MDO_t=MDO_t, OTH_t=OTH_t,
    WtW_HFO=WtW_HFO, WtW_LFO=WtW_LFO, WtW_MDO=WtW_MDO, WtW_OTH=WtW_OTH,
    LCV_HFO=LCV_HFO, LCV_LFO=LCV_LFO, LCV_MDO=LCV_MDO, LCV_OTH=LCV_OTH,
    PREMIUM=PREMIUM,
)

# ──────────────────────────────────────────────────────────────────────────────
# Calculate base metrics
# ──────────────────────────────────────────────────────────────────────────────
TOTAL_MJ = fi.total_MJ()
GFI = fi.gfi()

kpi1, kpi2 = st.columns(2)
kpi1.metric("GFI (gCO₂e/MJ)", f"{GFI:.3f}")
kpi2.metric("Total energy (MJ)", f"{TOTAL_MJ:,.0f}")

# GFI plot vs targets — step-wise (compact)
X_STEP = YEARS + [YEARS[-1] + 1]
base_step   = [GFI_BASE[y] for y in YEARS]   + [GFI_BASE[YEARS[-1]]]
direct_step = [GFI_DIRECT[y] for y in YEARS] + [GFI_DIRECT[YEARS[-1]]]
gfi_step    = [GFI] * len(X_STEP)
baseline_step = [GFI2008] * len(X_STEP)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=X_STEP, y=gfi_step, mode="lines", name="GFI attained",
    line=dict(width=2), line_shape="hv"
))
fig.add_trace(go.Scatter(
    x=X_STEP, y=base_step, mode="lines", name="Base target (step)",
    line=dict(dash="dash", width=2), line_shape="hv"
))
fig.add_trace(go.Scatter(
    x=X_STEP, y=direct_step, mode="lines", name="Direct target (step)",
    line=dict(dash="dot", width=2), line_shape="hv"
))
fig.add_trace(go.Scatter(
    x=X_STEP, y=baseline_step, mode="lines", name="Baseline 2008",
    line=dict(color="black", dash="longdash", width=2), line_shape="hv"
))
fig.update_layout(
    height=260,  # compact
    margin=dict(l=6, r=6, t=26, b=4),
    yaxis_title="gCO₂e/MJ",
    xaxis_title="Year",
    xaxis=dict(tickmode="array", tickvals=YEARS, tickfont=dict(size=10)),
    yaxis=dict(tickfont=dict(size=10)),
    legend=dict(orientation="h", y=-0.25)
)
st.plotly_chart(fig, use_container_width=True)

# Per-year tables and bars
rows: List[Dict] = []

# Dynamic reduction column name per selection
if reduce_choice == "HFO":
    red_col_name = "HFO_Reduction_For_Opt_Cost_t"
elif reduce_choice == "LFO":
    red_col_name = "LFO_Reduction_For_Opt_Cost_t"
else:
    red_col_name = "MDO/MGO_Reduction_For_Opt_Cost_t"

for yr in YEARS:
    deficit_t = deficit_surplus_tCO2eq(GFI, TOTAL_MJ, yr)
    t1_usd, t2_usd, ben_usd = tier_costs_usd(GFI, TOTAL_MJ, yr)

    # Optimization (only for reporting deltas; costs below are pre-optimization)
    sel_red_t, oth_inc_t, gfi_new, reg_cost_opt, premium_cost_opt = optimize_energy_neutral(
        fi, yr, reduce_fuel=reduce_choice
    )

    rows.append({
        "Year": yr,
        "GFI (g/MJ)": round(GFI, 6),
        "GFI_Deficit_Surplus_tCO2eq": deficit_t,
        "GFI_Tier_1_Cost_USD": t1_usd,
        "GFI_Tier_2_Cost_USD": t2_usd,
        "GFI_Benefit_USD": ben_usd,
        # Regulatory, Premium, Total based on INITIAL values (unchanged requirement)
        "Regulatory_Cost_USD": t1_usd + t2_usd + ben_usd,
        "Premium_Fuel_Cost_USD": PREMIUM * OTH_t,
        "Total_Cost_USD": (t1_usd + t2_usd + ben_usd) + (PREMIUM * OTH_t),
        # Optimization deltas LAST:
        red_col_name: sel_red_t,
        "Other_Fuel_Increase_For_Opt_Cost_t": oth_inc_t,
    })

res_df = pd.DataFrame(rows)

# Explicit column order so the optimization columns appear at the end
res_df = res_df[[
    "Year",
    "GFI (g/MJ)",
    "GFI_Deficit_Surplus_tCO2eq",
    "GFI_Tier_1_Cost_USD",
    "GFI_Tier_2_Cost_USD",
    "GFI_Benefit_USD",
    "Regulatory_Cost_USD",
    "Premium_Fuel_Cost_USD",
    "Total_Cost_USD",
    red_col_name,
    "Other_Fuel_Increase_For_Opt_Cost_t",
]]

st.subheader("Per-Year Results (2028–2035)")
st.dataframe(res_df.style.format({
    "GFI (g/MJ)": "{:.3f}",
    "GFI_Deficit_Surplus_tCO2eq": "{:.3f}",
    "GFI_Tier_1_Cost_USD": "{:,.0f}",
    "GFI_Tier_2_Cost_USD": "{:,.0f}",
    "GFI_Benefit_USD": "{:,.0f}",
    "Regulatory_Cost_USD": "{:,.0f}",
    "Premium_Fuel_Cost_USD": "{:,.0f}",
    "Total_Cost_USD": "{:,.0f}",
    red_col_name: "{:.3f}",
    "Other_Fuel_Increase_For_Opt_Cost_t": "{:.3f}",
}), use_container_width=True, height=360)

# Bar charts — compact with labels bigger & bold-like
def bar_chart(title: str, ycol: str):
    fmt_map = {
        "GFI_Deficit_Surplus_tCO2eq": ",.1f",
        "Regulatory_Cost_USD": ",.0f",
        "Premium_Fuel_Cost_USD": ",.0f",
        "Total_Cost_USD": ",.0f",
        "GFI_Tier_1_Cost_USD": ",.0f",
        "GFI_Tier_2_Cost_USD": ",.0f",
        "GFI_Benefit_USD": ",.0f",
    }
    textfmt = fmt_map.get(ycol, ",.2f")

    figb = px.bar(res_df, x="Year", y=ycol, title=title, text=ycol)
    figb.update_traces(
        texttemplate=f"%{{text:{textfmt}}}",
        textposition="outside",
        cliponaxis=False,
        outsidetextfont=dict(size=13, family="Arial Black")
    )
    figb.update_layout(
        height=210,
        margin=dict(l=4, r=4, t=24, b=4),
        bargap=0.15,
        bargroupgap=0.05,
        showlegend=False,
        xaxis=dict(tickmode="array", tickvals=YEARS, tickfont=dict(size=10)),
        yaxis=dict(title=None, tickfont=dict(size=10)),
        uniformtext_minsize=9,
        uniformtext_mode="hide"
    )

    yvals = res_df[ycol].astype(float)
    if not yvals.empty:
        ymax = float(yvals.max())
        ymin = float(yvals.min())
        pad_up = 0.10 * abs(ymax) if ymax != 0 else 1.0
        pad_dn = 0.10 * abs(ymin) if ymin != 0 else 0.0
        if ymax != ymin:
            figb.update_yaxes(range=[ymin - pad_dn, ymax + pad_up])

    st.plotly_chart(figb, use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    bar_chart("GFI Deficit/Surplus [tCO₂e]", "GFI_Deficit_Surplus_tCO2eq")
with c2:
    bar_chart("Regulatory Cost [USD]", "Regulatory_Cost_USD")

c3, c4 = st.columns(2)
with c3:
    bar_chart("Premium Fuel Cost [USD]", "Premium_Fuel_Cost_USD")
with c4:
    bar_chart("Total Cost [USD]", "Total_Cost_USD")

c5, c6 = st.columns(2)
with c5:
    bar_chart("Tier 1 Cost [USD]", "GFI_Tier_1_Cost_USD")
with c6:
    bar_chart("Tier 2 Cost [USD]", "GFI_Tier_2_Cost_USD")

bar_chart("Benefit [USD] (negative = credit)", "GFI_Benefit_USD")

# Download Excel
buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as xw:
    pd.DataFrame({
        "Parameter": [
            "HFO_t","LFO_t","MDO_t","OTH_t",
            "WtW_HFO","WtW_LFO","WtW_MDO","WtW_OTH",
            "LCV_HFO","LCV_LFO","LCV_MDO","LCV_OTH",
            "Premium (Others − Selected Fuel)","Selected fuel to reduce"
        ],
        "Value": [
            HFO_t,LFO_t,MDO_t,OTH_t,
            WtW_HFO,WtW_LFO,WtW_MDO,WtW_OTH,
            LCV_HFO,LCV_LFO,LCV_MDO,LCV_OTH,
            PREMIUM, reduce_choice
        ],
        "Units": [
            "t","t","t","t",
            "g/MJ","g/MJ","g/MJ","g/MJ",
            "MJ/t","MJ/t","MJ/t","MJ/t",
            "USD/t","—"
        ],
    }).to_excel(xw, sheet_name="Inputs", index=False)
    res_df.to_excel(xw, sheet_name="Results_2028_2035", index=False)

st.download_button(
    label="Download results (Excel)",
    data=buf.getvalue(),
    file_name="GFI_Bunkering_Optimizer_SingleVessel.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.caption("© 2025 — Single-vessel GFI optimizer. Logic and outputs remain as in the original; only the optimization now allows selecting HFO/LFO/MDO-MGO as the reduced fuel.")
