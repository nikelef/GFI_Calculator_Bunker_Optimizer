# app.py — GFI Bunkering Optimizer
# ------------------------------------------------
# Single-vessel Streamlit app with energy-neutral optimization (closed-form).
# Units:
#   Mass: tons
#   LCV : MJ/ton
#   WtW : gCO2eq/MJ

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
# Constants (defaults)
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

# Cost rates (USD per tCO2eq) — will be overridden by sidebar inputs
TIER1_COST = 100.0
TIER2_COST = 380.0
BENEFIT_RATE = 190.0  # negative mass → negative $ (credit)

# Defaults persistence file
DEFAULTS_PATH = ".gfi_bunkering_defaults.json"

# Labels (for UI)
CF_LABELS = {
    "HFO": "HFO",
    "LFO": "LFO",
    "MDO": "MDO/MGO",
    "BIO": "BIO",
}

# ──────────────────────────────────────────────────────────────────────────────
# Data & defaults
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class FuelInputs:
    # Masses [t]
    HFO_t: float
    LFO_t: float
    MDO_t: float
    BIO_t: float

    # WtW [gCO2eq/MJ]
    WtW_HFO: float
    WtW_LFO: float
    WtW_MDO: float
    WtW_BIO: float

    # LCV [MJ/t]
    LCV_HFO: float
    LCV_LFO: float
    LCV_MDO: float
    LCV_BIO: float

    # Premium USD/t (Bio − Selected Fuel)
    PREMIUM: float

    def total_MJ(self) -> float:
        return (
            self.HFO_t * self.LCV_HFO
            + self.LFO_t * self.LCV_LFO
            + self.MDO_t * self.LCV_MDO
            + self.BIO_t * self.LCV_BIO
        )

    def gfi(self) -> float:
        denom = self.total_MJ()
        if denom <= 0:
            return 0.0
        num_g = (
            self.HFO_t * self.LCV_HFO * self.WtW_HFO
            + self.LFO_t * self.LCV_LFO * self.WtW_LFO
            + self.MDO_t * self.LCV_MDO * self.WtW_MDO
            + self.BIO_t * self.LCV_BIO * self.WtW_BIO
        )
        return num_g / denom  # gCO2eq/MJ


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
# Core calculations (defined BEFORE any usage)
# ──────────────────────────────────────────────────────────────────────────────
def deficit_surplus_tCO2eq(gfi_g_per_MJ: float, total_MJ: float, year: int) -> float:
    """GFI_Deficit_Surplus_year in tCO2eq. Positive = deficit, Negative = surplus."""
    if total_MJ <= 0:
        return 0.0

    base = GFI_BASE[year]
    direct = GFI_DIRECT[year]

    if gfi_g_per_MJ > base:
        g_g = (gfi_g_per_MJ - direct) * total_MJ
    elif gfi_g_per_MJ >= direct:
        g_g = (gfi_g_per_MJ - direct) * total_MJ
    else:
        g_g = (gfi_g_per_MJ - direct) * total_MJ  # negative surplus

    return g_g / 1e6  # grams → tonnes


def tier_costs_usd(gfi_g_per_MJ: float, total_MJ: float, year: int) -> Tuple[float, float, float]:
    """Return (Tier1 USD, Tier2 USD, Benefit USD). Benefit negative below Direct."""
    if total_MJ <= 0:
        return 0.0, 0.0, 0.0

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

    t1_usd = tier1_mt * TIER1_COST
    t2_usd = tier2_mt * TIER2_COST
    ben_usd = benefit_mt * BENEFIT_RATE
    return t1_usd, t2_usd, ben_usd


def optimize_energy_neutral(
    fi: FuelInputs,
    year: int,
    reduce_fuel: str = "HFO",  # "HFO" | "LFO" | "MDO/MGO"
    coarse_steps: int = 200,    # accepted for compatibility (unused)
    fine_window: float = 0.02,  # accepted for compatibility (unused)
    fine_step: float = 0.001,   # accepted for compatibility (unused)
) -> Tuple[float, float, float, float, float]:
    """
    Closed-form optimizer: TotalCost(f) is piecewise-linear in f with kinks only at
    GFI == Direct_y and GFI == Base_y. Therefore the global minimum lies in
    {0, f_direct, f_base, 1}. Returns:
    (selected_fuel_reduction_t, bio_increase_t, gfi_new, reg_cost_usd, premium_cost_usd)
    """
    # Map selected fuel
    if reduce_fuel == "HFO":
        sel_mass0, sel_lcv, sel_wtw = fi.HFO_t, fi.LCV_HFO, fi.WtW_HFO
    elif reduce_fuel == "LFO":
        sel_mass0, sel_lcv, sel_wtw = fi.LFO_t, fi.LCV_LFO, fi.WtW_LFO
    else:  # "MDO/MGO"
        sel_mass0, sel_lcv, sel_wtw = fi.MDO_t, fi.LCV_MDO, fi.WtW_MDO

    D0 = fi.total_MJ()
    if sel_mass0 <= 0 or fi.LCV_BIO <= 0 or D0 <= 0:
        return 0.0, 0.0, fi.gfi(), 0.0, 0.0

    # GFI(f) = G0 + s*f
    G0 = fi.gfi()
    s = (sel_mass0 * sel_lcv / D0) * (fi.WtW_BIO - sel_wtw)

    def eval_total(f: float) -> Tuple[float, float, float, float, float]:
        # New masses after reducing selected fuel by fraction f and increasing BIO for energy neutrality
        sel_new = sel_mass0 * (1.0 - f)
        d_sel = sel_mass0 - sel_new
        bio_new = fi.BIO_t + (d_sel * sel_lcv) / fi.LCV_BIO

        hfo_new, lfo_new, mdo_new = fi.HFO_t, fi.LFO_t, fi.MDO_t
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
            + bio_new * fi.LCV_BIO
        )

        if total_MJ <= 0:
            return np.inf, 0.0, 0.0, 0.0, 0.0

        num_g = (
            hfo_new * fi.LCV_HFO * fi.WtW_HFO
            + lfo_new * fi.LCV_LFO * fi.WtW_LFO
            + mdo_new * fi.LCV_MDO * fi.WtW_MDO
            + bio_new * fi.LCV_BIO * fi.WtW_BIO
        )
        gfi_new = num_g / total_MJ

        t1, t2, ben = tier_costs_usd(gfi_new, total_MJ, year)
        reg_cost = t1 + t2 + ben
        premium_cost = max(bio_new - fi.BIO_t, 0.0) * fi.PREMIUM
        total_cost = reg_cost + premium_cost
        return total_cost, gfi_new, reg_cost, premium_cost, d_sel

    # Candidate fractions
    candidates: List[float] = [0.0, 1.0]
    if abs(s) > 0:
        f_direct = (GFI_DIRECT[year] - G0) / s
        f_base = (GFI_BASE[year] - G0) / s
        if 0.0 <= f_direct <= 1.0:
            candidates.append(float(f_direct))
        if 0.0 <= f_base <= 1.0:
            candidates.append(float(f_base))

    best = None
    for f in candidates:
        tot, gfi_new, reg, prem, d_sel = eval_total(f)
        if (best is None) or (tot < best[0] - 1e-12):
            best = (tot, f, gfi_new, reg, prem, d_sel)

    _, f_best, gfi_best, reg_best, prem_best, d_sel_best = best
    sel_red = d_sel_best
    bio_inc = (sel_red * sel_lcv) / fi.LCV_BIO if fi.LCV_BIO > 0 else 0.0
    return sel_red, bio_inc, gfi_best, reg_best, prem_best


# ──────────────────────────────────────────────────────────────────────────────
# US-formatted numeric input helpers (commas + exactly two decimals) in SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
def _format_state_number(key: str, min_value: float = 0.0) -> None:
    """Normalize st.session_state[key] to US format with two decimals."""
    s = st.session_state.get(key, "")
    s = (s or "").strip().replace(" ", "").replace(",", "")
    try:
        v = float(s)
    except Exception:
        v = float(st.session_state.get(f"__prev_{key}", min_value))
    v = max(v, min_value)
    st.session_state[f"__prev_{key}"] = v
    st.session_state[key] = f"{v:,.2f}"

def us_number_input(label: str, default: float, key: str, *, container, min_value: float = 0.0) -> float:
    """
    Text input *inside the provided container* (e.g., st.sidebar columns) that shows
    numbers as US formatted (1,234.56) with 2 decimals, while returning a clean float.
    """
    if key not in st.session_state:
        st.session_state[key] = f"{float(default):,.2f}"
        st.session_state[f"__prev_{key}"] = float(default)
    container.text_input(label, key=key, on_change=_format_state_number, args=(key, min_value))
    try:
        return max(float(st.session_state[key].replace(",", "")), min_value)
    except Exception:
        return max(float(st.session_state.get(f"__prev_{key}", default)), min_value)

# ──────────────────────────────────────────────────────────────────────────────
# UI — Streamlit
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="IMO GFI Calculator - Bunkering Optimizer", layout="wide")
st.title("IMO GFI Calculator - Bunkering Optimizer")

# Make the sidebar (input column) a bit wider
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { width: 420px; min-width: 420px; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.expander("Methodology & Units", expanded=False):
    st.markdown(
        r"""
**Formulas:**

- **GFI** \[gCO₂e/MJ] = {sum_i m_i * LCV_i * WtW_i}/ {sum_i m_i * LCV_i)
- **Deficit/Surplus** \[tCO₂e] for year *y*:
  - If \(GFI > Base_y\): \((GFI−Direct_y)\cdot TotalMJ / 10^6\)
  - If \(Direct_y \le GFI \le Base_y\): \((GFI−Direct_y)\cdot TotalMJ / 10^6\)
  - If \(GFI < Direct_y\): \((GFI−Direct_y)\cdot TotalMJ / 10^6\) (negative surplus)
- **Tier costs default values** \[USD per tCO2eq]: Tier-1 = 100, Tier-2 = 380, Benefit = 190 (when below direct limit)
- **Optimization (per year)**: reduce **selected fuel (HFO/LFO/MDO-MGO)** by Δt and
  increase **BIO** (or any other suitable fuel) by Δt·LCV_sel/LCV_BIO (energy-neutral). Objective:
  minimize (Tier1 + Tier2 + Benefit + Premium)

**Units**: Mass in tons; LCV in MJ/ton; WtW in gCO₂e/MJ.
"""
    )

# Load persisted defaults
states = load_defaults()

# Sidebar inputs (left), US-formatted in place; sidebar widened
st.sidebar.header("Inputs")

colA, colB = st.sidebar.columns(2)
HFO_t = us_number_input(f"{CF_LABELS['HFO']} (tons)", float(states.get("HFO_t", 100.0)), key="inp_HFO_t", container=colA)
LFO_t = us_number_input(f"{CF_LABELS['LFO']} (tons)", float(states.get("LFO_t", 0.0)), key="inp_LFO_t", container=colB)
MDO_t = us_number_input(f"{CF_LABELS['MDO']} (tons)", float(states.get("MDO_t", 0.0)), key="inp_MDO_t", container=colA)
BIO_t = us_number_input(f"{CF_LABELS['BIO']} (tons)", float(states.get("BIO_t", 0.0)), key="inp_BIO_t", container=colB)

st.sidebar.markdown("---")
colC, colD = st.sidebar.columns(2)
WtW_HFO = us_number_input("WtW HFO [gCO₂e/MJ]", float(states.get("WtW_HFO", 92.784)), key="inp_WtW_HFO", container=colC)
WtW_LFO = us_number_input("WtW LFO [gCO₂e/MJ]", float(states.get("WtW_LFO", 91.251)), key="inp_WtW_LFO", container=colD)
WtW_MDO = us_number_input("WtW MDO/MGO [gCO₂e/MJ]", float(states.get("WtW_MDO", 93.932)), key="inp_WtW_MDO", container=colC)
WtW_BIO = us_number_input("WtW BIO [gCO₂e/MJ]", float(states.get("WtW_BIO", 70.366)), key="inp_WtW_BIO", container=colD)

st.sidebar.markdown("---")
colE, colF = st.sidebar.columns(2)
LCV_HFO = us_number_input("LCV HFO [MJ/ton]", float(states.get("LCV_HFO", 40200.0)), key="inp_LCV_HFO", container=colE)
LCV_LFO = us_number_input("LCV LFO [MJ/ton]", float(states.get("LCV_LFO", 41000.0)), key="inp_LCV_LFO", container=colF)
LCV_MDO = us_number_input("LCV MDO/MGO [MJ/ton]", float(states.get("LCV_MDO", 42700.0)), key="inp_LCV_MDO", container=colE)
LCV_BIO = us_number_input("LCV BIO [MJ/ton]", float(states.get("LCV_BIO", 37000.0)), key="inp_LCV_BIO", container=colF)

# Fuel to reduce selector
st.sidebar.markdown("---")
reduce_choice = st.sidebar.selectbox(
    "Fuel to reduce (for optimization). BIO will replace it considering constant energy",
    options=["HFO", "LFO", "MDO/MGO"],
    index=int(states.get("reduce_idx", 0))
)

# Cost rates inputs (below LCVs, above Premium)
st.sidebar.markdown("**Cost rates [USD per tCO₂e]**")
colG, colH, colI = st.sidebar.columns(3)
TIER1_COST = us_number_input("Tier 1", float(states.get("TIER1_COST", 100.0)), key="inp_TIER1_COST", container=colG, min_value=0.0)
TIER2_COST = us_number_input("Tier 2", float(states.get("TIER2_COST", 380.0)), key="inp_TIER2_COST", container=colH, min_value=0.0)
BENEFIT_RATE = us_number_input("Benefit", float(states.get("BENEFIT_RATE", 190.0)), key="inp_BENEFIT_RATE", container=colI, min_value=0.0)

# Premium (placed after cost rates)
st.sidebar.markdown("---")
PREMIUM = us_number_input(
    f"Premium [USD/ton] (Biofuel Cost - {reduce_choice} cost)",
    float(states.get("PREMIUM", 305.0)),
    key="inp_PREMIUM",
    container=st.sidebar,
    min_value=0.0
)

# Save defaults button
if st.sidebar.button("Save as defaults", use_container_width=True):
    new_states = {
        "HFO_t": HFO_t, "LFO_t": LFO_t, "MDO_t": MDO_t, "BIO_t": BIO_t,
        "WtW_HFO": WtW_HFO, "WtW_LFO": WtW_LFO, "WtW_MDO": WtW_MDO, "WtW_BIO": WtW_BIO,
        "LCV_HFO": LCV_HFO, "LCV_LFO": LCV_LFO, "LCV_MDO": LCV_MDO, "LCV_BIO": LCV_BIO,
        "TIER1_COST": TIER1_COST, "TIER2_COST": TIER2_COST, "BENEFIT_RATE": BENEFIT_RATE,
        "PREMIUM": PREMIUM, "reduce_idx": ["HFO", "LFO", "MDO/MGO"].index(reduce_choice)
    }
    save_defaults(new_states)
    st.sidebar.success("Defaults saved.")

# ──────────────────────────────────────────────────────────────────────────────
# Base metrics
# ──────────────────────────────────────────────────────────────────────────────
fi = FuelInputs(
    HFO_t=HFO_t, LFO_t=LFO_t, MDO_t=MDO_t, BIO_t=BIO_t,
    WtW_HFO=WtW_HFO, WtW_LFO=WtW_LFO, WtW_MDO=WtW_MDO, WtW_BIO=WtW_BIO,
    LCV_HFO=LCV_HFO, LCV_LFO=LCV_LFO, LCV_MDO=LCV_MDO, LCV_BIO=LCV_BIO,
    PREMIUM=PREMIUM,
)

TOTAL_MJ = fi.total_MJ()
GFI = fi.gfi()

kpi1, kpi2 = st.columns(2)
kpi1.metric("GFI (gCO₂e/MJ)", f"{GFI:,.2f}")
kpi2.metric("Total energy (MJ)", f"{TOTAL_MJ:,.2f}")

# Step-wise targets plot (with unified hover showing all traces)
X_STEP = YEARS + [YEARS[-1] + 1]
base_step = [GFI_BASE[y] for y in YEARS] + [GFI_BASE[YEARS[-1]]]
direct_step = [GFI_DIRECT[y] for y in YEARS] + [GFI_DIRECT[YEARS[-1]]]
gfi_step = [GFI] * len(X_STEP)
baseline_step = [GFI2008] * len(X_STEP)

fig = go.Figure()

# Order traces top→bottom in hover: Baseline (top), GFI attained, Base target, Direct target (bottom)
fig.add_trace(go.Scatter(
    x=X_STEP, y=baseline_step, mode="lines", name="Baseline 2008",
    line=dict(dash="longdash", width=2, color="black"), line_shape="hv",
    hovertemplate="Baseline 2008: %{y:,.2f} gCO₂e/MJ<extra></extra>"
))
fig.add_trace(go.Scatter(
    x=X_STEP, y=gfi_step, mode="lines", name="GFI attained",
    line=dict(width=2), line_shape="hv",
    hovertemplate="GFI attained: %{y:,.2f} gCO₂e/MJ<extra></extra>"
))
fig.add_trace(go.Scatter(
    x=X_STEP, y=base_step, mode="lines", name="Base target (step)",
    line=dict(dash="dash", width=2), line_shape="hv",
    hovertemplate="Base target: %{y:,.2f} gCO₂e/MJ<extra></extra>"
))
fig.add_trace(go.Scatter(
    x=X_STEP, y=direct_step, mode="lines", name="Direct target (step)",
    line=dict(dash="dot", width=2), line_shape="hv",
    hovertemplate="Direct target: %{y:,.2f} gCO₂e/MJ<extra></extra>"
))

fig.update_layout(
    height=260,
    margin=dict(l=6, r=6, t=26, b=4),
    yaxis_title="gCO₂e/MJ",
    xaxis_title="Year",
    xaxis=dict(tickmode="array", tickvals=YEARS, tickfont=dict(size=10)),
    yaxis=dict(tickfont=dict(size=10)),
    legend=dict(orientation="h", y=-0.25),
    hovermode="x unified",
    hoverlabel=dict(align="left", namelength=-1)
)
fig.update_yaxes(tickformat=",.2f")
st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Per-year calculations & optimization (deltas reported)
# ──────────────────────────────────────────────────────────────────────────────
rows: List[Dict] = []

# Dynamic reduction column name
if reduce_choice == "HFO":
    red_col_name = "HFO_Reduction_For_Opt_Cost_t"
elif reduce_choice == "LFO":
    red_col_name = "LFO_Reduction_For_Opt_Cost_t"
else:
    red_col_name = "MDO/MGO_Reduction_For_Opt_Cost_t"

for yr in YEARS:
    if TOTAL_MJ <= 0:
        deficit_t = 0.0
        t1_usd = t2_usd = ben_usd = 0.0
        sel_red_t = bio_inc_t = 0.0
        tot_cost_opt = 0.0
    else:
        deficit_t = deficit_surplus_tCO2eq(GFI, TOTAL_MJ, yr)
        t1_usd, t2_usd, ben_usd = tier_costs_usd(GFI, TOTAL_MJ, yr)
        sel_red_t, bio_inc_t, gfi_new, reg_cost_opt, premium_cost_opt = optimize_energy_neutral(
            fi, yr, reduce_fuel=reduce_choice
        )
        tot_cost_opt = reg_cost_opt + premium_cost_opt  # final optimized cost

    rows.append({
        "Year": yr,
        "GFI (g/MJ)": round(GFI, 6),
        "GFI_Deficit_Surplus_tCO2eq": deficit_t,
        "GFI_Tier_1_Cost_USD": t1_usd,
        "GFI_Tier_2_Cost_USD": t2_usd,
        "GFI_Benefit_USD": ben_usd,
        # Regulatory, Premium, Total based on INITIAL values
        "Regulatory_Cost_USD": t1_usd + t2_usd + ben_usd,
        "Premium_Fuel_Cost_USD": PREMIUM * BIO_t,
        "Total_Cost_USD": (t1_usd + t2_usd + ben_usd) + (PREMIUM * BIO_t),
        # Optimization outputs
        "Total_Cost_USD_Opt": tot_cost_opt,
        red_col_name: sel_red_t,
        "Bio_Fuel_Increase_For_Opt_Cost_t": bio_inc_t,
    })

# ──────────────────────────────────────────────────────────────────────────────
# Per-year results table (place Total_Cost_USD_Opt at the END)
# ──────────────────────────────────────────────────────────────────────────────
res_df = pd.DataFrame(rows)
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
    "Bio_Fuel_Increase_For_Opt_Cost_t",
    "Total_Cost_USD_Opt",  # last
]]

st.subheader("Per-Year Results (2028–2035)")
st.dataframe(
    res_df.style.format({
        "GFI (g/MJ)": "{:,.2f}",
        "GFI_Deficit_Surplus_tCO2eq": "{:,.2f}",
        "GFI_Tier_1_Cost_USD": "{:,.2f}",
        "GFI_Tier_2_Cost_USD": "{:,.2f}",
        "GFI_Benefit_USD": "{:,.2f}",
        "Regulatory_Cost_USD": "{:,.2f}",
        "Premium_Fuel_Cost_USD": "{:,.2f}",
        "Total_Cost_USD": "{:,.2f}",
        red_col_name: "{:,.2f}",
        "Bio_Fuel_Increase_For_Opt_Cost_t": "{:,.2f}",
        "Total_Cost_USD_Opt": "{:,.2f}",
    }),
    use_container_width=True, height=360
)

# ──────────────────────────────────────────────────────────────────────────────
# Compact bar charts (requested order)
# ──────────────────────────────────────────────────────────────────────────────
def bar_chart(title: str, ycol: str):
    fmt_map = {
        "GFI_Deficit_Surplus_tCO2eq": ",.2f",
        "Regulatory_Cost_USD": ",.2f",
        "Premium_Fuel_Cost_USD": ",.2f",
        "Total_Cost_USD": ",.2f",
        "Total_Cost_USD_Opt": ",.2f",
        "GFI_Tier_1_Cost_USD": ",.2f",
        "GFI_Tier_2_Cost_USD": ",.2f",
        "GFI_Benefit_USD": ",.2f",
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
    figb.update_yaxes(tickformat=",.2f")

    yvals = res_df[ycol].astype(float)
    if not yvals.empty:
        ymax = float(yvals.max())
        ymin = float(yvals.min())
        pad_up = 0.10 * abs(ymax) if ymax != 0 else 1.0
        pad_dn = 0.10 * abs(ymin) if ymin != 0 else 0.0
        if ymax != ymin:
            figb.update_yaxes(range=[ymin - pad_dn, ymax + pad_up])

    st.plotly_chart(figb, use_container_width=True)

# Core charts in two-column rows
c1, c2 = st.columns(2)
with c1:
    bar_chart("GFI Deficit/Surplus [tCO₂e]", "GFI_Deficit_Surplus_tCO2eq")
with c2:
    bar_chart("Regulatory Cost [USD]", "Regulatory_Cost_USD")

c3, c4 = st.columns(2)
with c3:
    bar_chart("Premium Fuel Cost [USD]", "Premium_Fuel_Cost_USD")
with c4:
    bar_chart("Tier 1 Cost [USD]", "GFI_Tier_1_Cost_USD")

c5, c6 = st.columns(2)
with c5:
    bar_chart("Tier 2 Cost [USD]", "GFI_Tier_2_Cost_USD")

# Totals at the end (above Benefit)
bar_chart("Total Cost [USD]", "Total_Cost_USD")
bar_chart("Total Cost (Optimized) [USD]", "Total_Cost_USD_Opt")

# Benefit last
bar_chart("Benefit [USD] (negative = credit)", "GFI_Benefit_USD")

# ──────────────────────────────────────────────────────────────────────────────
# Download Excel
# ──────────────────────────────────────────────────────────────────────────────
buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as xw:
    pd.DataFrame({
        "Parameter": [
            "HFO_t","LFO_t","MDO_t","BIO_t",
            "WtW_HFO","WtW_LFO","WtW_MDO","WtW_BIO",
            "LCV_HFO","LCV_LFO","LCV_MDO","LCV_BIO",
            "Tier 1 Cost (USD/tCO2e)","Tier 2 Cost (USD/tCO2e)","Benefit Rate (USD/tCO2e)",
            "Premium (Bio − Selected Fuel)","Selected fuel to reduce"
        ],
        "Value": [
            HFO_t,LFO_t,MDO_t,BIO_t,
            WtW_HFO,WtW_LFO,WtW_MDO,WtW_BIO,
            LCV_HFO,LCV_LFO,LCV_MDO,LCV_BIO,
            TIER1_COST, TIER2_COST, BENEFIT_RATE,
            PREMIUM, reduce_choice
        ],
        "Units": [
            "t","t","t","t",
            "g/MJ","g/MJ","g/MJ","g/MJ",
            "MJ/t","MJ/t","MJ/t","MJ/t",
            "USD/tCO2e","USD/tCO2e","USD/tCO2e",
            "USD/t","—"
        ],
    }).to_excel(xw, sheet_name="Inputs", index=False)
    res_df.to_excel(xw, sheet_name="Results_2028_2035", index=False)

st.download_button(
    label="Download results (Excel)",
    data=buf.getvalue(),
    file_name="GFI_Bunkering_Optimizer_1_1.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.caption("© 2025 — Single-vessel GFI optimizer. Initial costs shown; optimization reports deltas and Total_Cost_USD_Opt.")
