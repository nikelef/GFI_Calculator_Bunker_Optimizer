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
# Constants
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
# Data & utils
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
# Core calculations
# ──────────────────────────────────────────────────────────────────────────────
def deficit_surplus_tCO2eq(gfi_g_per_MJ: float, total_MJ: float, year: int) -> float:
    """GFI_Deficit_Surplus_year in tCO2eq. Positive = deficit, Negative = surplus."""
    if total_MJ <= 0:
        return 0.0

    base = GFI_BASE[year]
    direct = GFI_DIRECT[year]

    if gfi_g_per_MJ > base:
        # (GFI−Base + Base−Direct) * MJ
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
# UI — Streamlit
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="GFI Calculator - Bunkering Optimizer 1.1 - TMS DRY", layout="wide")
st.title("GFI Calculator - Bunkering Optimizer 1.1 -  TMS DRY")

with st.expander("Methodology & Units", expanded=False):
    st.markdown(
        r"""
**Formulas:**

- **GFI** \[gCO₂e/MJ] = \(\frac{\sum_i m_i \cdot LCV_i \cdot WtW_i}{\sum_i m_i \cdot LCV_i}\)
- **Deficit/Surplus** \[tCO₂e] for year *y*:
  - If \(GFI > Base_y\): \((GFI−Direct_y)\cdot TotalMJ / 10^6\)
  - If \(Direct_y \le GFI \le Base_y\): \((GFI−Direct_y)\cdot TotalMJ / 10^6\)
  - If \(GFI < Direct_y\): \((GFI−Direct_y)\cdot TotalMJ / 10^6\) (negative surplus)
- **Tier costs** \[USD]: Tier-1 = 100, Tier-2 = 380, Benefit = 190 × (negative mass)
- **Optimization (per year)**: reduce **selected fuel (HFO/LFO/MDO-MGO)** by Δt and
  increase **BIO** by Δt·LCV_sel/LCV_BIO (energy-neutral). Objective:
  minimize \(Tier1 + Tier2 + Benefit + Premium \cdot \max(\Delta BIO,0)\).

**Units**: Mass in tons; LCV in MJ/ton; WtW in gCO₂e/MJ.
"""
    )

# Load persisted defaults
states = load_defaults()

# Sidebar inputs
st.sidebar.header("Inputs")

colA, colB = st.sidebar.columns(2)
HFO_t = colA.number_input(f"{CF_LABELS['HFO']} (tons)", min_value=0.0, value=float(states.get("HFO_t", 100.0)), step=0.1)
LFO_t = colB.number_input(f"{CF_LABELS['LFO']} (tons)", min_value=0.0, value=float(states.get("LFO_t", 0.0)), step=0.1)
MDO_t = colA.number_input(f"{CF_LABELS['MDO']} (tons)", min_value=0.0, value=float(states.get("MDO_t", 0.0)), step=0.1)
BIO_t = colB.number_input(f"{CF_LABELS['BIO']} (tons)", min_value=0.0, value=float(states.get("BIO_t", 0.0)), step=0.1)

st.sidebar.markdown("---")
colC, colD = st.sidebar.columns(2)
WtW_HFO = colC.number_input("WtW HFO [gCO₂e/MJ]", min_value=0.0, value=float(states.get("WtW_HFO", 92.784)), step=0.001, format="%.3f")
WtW_LFO = colD.number_input("WtW LFO [gCO₂e/MJ]", min_value=0.0, value=float(states.get("WtW_LFO", 91.251)), step=0.001, format="%.3f")
WtW_MDO = colC.number_input("WtW MDO/MGO [gCO₂e/MJ]", min_value=0.0, value=float(states.get("WtW_MDO", 93.932)), step=0.001, format="%.3f")
WtW_BIO = colD.number_input("WtW BIO [gCO₂e/MJ]", min_value=0.0, value=float(states.get("WtW_BIO", 70.366)), step=0.001, format="%.3f")

st.sidebar.markdown("---")
colE, colF = st.sidebar.columns(2)
LCV_HFO = colE.number_input("LCV HFO [MJ/ton]", min_value=0.0, value=float(states.get("LCV_HFO", 40200.0)), step=100.0)
LCV_LFO = colF.number_input("LCV LFO [MJ/ton]", min_value=0.0, value=float(states.get("LCV_LFO", 41000.0)), step=100.0)
LCV_MDO = colE.number_input("LCV MDO/MGO [MJ/ton]", min_value=0.0, value=float(states.get("LCV_MDO", 42700.0)), step=100.0)
LCV_BIO = colF.number_input("LCV BIO [MJ/ton]", min_value=0.0, value=float(states.get("LCV_BIO", 37000.0)), step=100.0)

st.sidebar.markdown("---")
reduce_choice = st.sidebar.selectbox(
    "Fuel to reduce (for optimization)",
    options=["HFO", "LFO", "MDO/MGO"],
    index=int(states.get("reduce_idx", 0))
)

PREMIUM = st.sidebar.number_input(
    f"Premium [USD/ton] (Biofuel − {reduce_choice})",
    min_value=0.0,
    value=float(states.get("PREMIUM", 305.0)),
    step=10.0
)

if st.sidebar.button("Save as defaults", use_container_width=True):
    new_states = {
        "HFO_t": HFO_t, "LFO_t": LFO_t, "MDO_t": MDO_t, "BIO_t": BIO_t,
        "WtW_HFO": WtW_HFO, "WtW_LFO": WtW_LFO, "WtW_MDO": WtW_MDO, "WtW_BIO": WtW_BIO,
        "LCV_HFO": LCV_HFO, "LCV_LFO": LCV_LFO, "LCV_MDO": LCV_MDO, "LCV_BIO": LCV_BIO,
        "PREMIUM": PREMIUM, "reduce_idx": ["HFO", "LFO", "MDO/MGO"].index(reduce_choice)
    }
    save_defaults(new_states)
    st.sidebar.success("Defaults saved.")

# Build inputs
fi = FuelInputs(
    HFO_t=HFO_t, LFO_t=LFO_t, MDO_t=MDO_t, BIO_t=BIO_t,
    WtW_HFO=WtW_HFO, WtW_LFO=WtW_LFO, WtW_MDO=WtW_MDO, WtW_BIO=WtW_BIO,
    LCV_HFO=LCV_HFO, LCV_LFO=LCV_LFO, LCV_MDO=LCV_MDO, LCV_BIO=LCV_BIO,
    PREMIUM=PREMIUM,
)

# ──────────────────────────────────────────────────────────────────────────────
# Base metrics
# ──────────────────────────────────────────────────────────────────────────────
TOTAL_MJ = fi.total_MJ()
GFI = fi.gfi()

kpi1, kpi2 = st.columns(2)
kpi1.metric("GFI (gCO₂e/MJ)", f"{GFI:.3f}")
kpi2.metric("Total energy (MJ)", f"{TOTAL_MJ:,.0f}")

# Step-wise targets plot
X_STEP = YEARS + [YEARS[-1] + 1]
base_step = [GFI_BASE[y] for y in YEARS] + [GFI_BASE[YEARS[-1]]]
direct_step = [GFI_DIRECT[y] for y in YEARS] + [GFI_DIRECT[YEARS[-1]]]
gfi_step = [GFI] * len(X_STEP)
baseline_step = [GFI2008] * len(X_STEP)

fig = go.Figure()
fig.add_trace(go.Scatter(x=X_STEP, y=gfi_step, mode="lines", name="GFI attained", line=dict(width=2), line_shape="hv"))
fig.add_trace(go.Scatter(x=X_STEP, y=base_step, mode="lines", name="Base target (step)", line=dict(dash="dash", width=2), line_shape="hv"))
fig.add_trace(go.Scatter(x=X_STEP, y=direct_step, mode="lines", name="Direct target (step)", line=dict(dash="dot", width=2), line_shape="hv"))
fig.add_trace(go.Scatter(x=X_STEP, y=baseline_step, mode="lines", name="Baseline 2008", line=dict(color="black", dash="longdash", width=2), line_shape="hv"))
fig.update_layout(
    height=260,
    margin=dict(l=6, r=6, t=26, b=4),
    yaxis_title="gCO₂e/MJ",
    xaxis_title="Year",
    xaxis=dict(tickmode="array", tickvals=YEARS, tickfont=dict(size=10)),
    yaxis=dict(tickfont=dict(size=10)),
    legend=dict(orientation="h", y=-0.25)
)
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
    else:
        deficit_t = deficit_surplus_tCO2eq(GFI, TOTAL_MJ, yr)
        t1_usd, t2_usd, ben_usd = tier_costs_usd(GFI, TOTAL_MJ, yr)
        sel_red_t, bio_inc_t, gfi_new, reg_cost_opt, premium_cost_opt = optimize_energy_neutral(
            fi, yr, reduce_fuel=reduce_choice
        )

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
        # Optimization deltas at the end
        red_col_name: sel_red_t,
        "Bio_Fuel_Increase_For_Opt_Cost_t": bio_inc_t,
    })

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
]]

st.subheader("Per-Year Results (2028–2035)")
st.dataframe(
    res_df.style.format({
        "GFI (g/MJ)": "{:.3f}",
        "GFI_Deficit_Surplus_tCO2eq": "{:.3f}",
        "GFI_Tier_1_Cost_USD": "{:,.0f}",
        "GFI_Tier_2_Cost_USD": "{:,.0f}",
        "GFI_Benefit_USD": "{:,.0f}",
        "Regulatory_Cost_USD": "{:,.0f}",
        "Premium_Fuel_Cost_USD": "{:,.0f}",
        "Total_Cost_USD": "{:,.0f}",
        red_col_name: "{:.3f}",
        "Bio_Fuel_Increase_For_Opt_Cost_t": "{:.3f}",
    }),
    use_container_width=True, height=360
)

# ──────────────────────────────────────────────────────────────────────────────
# Compact bar charts
# ──────────────────────────────────────────────────────────────────────────────
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
            "Premium (Bio − Selected Fuel)","Selected fuel to reduce"
        ],
        "Value": [
            HFO_t,LFO_t,MDO_t,BIO_t,
            WtW_HFO,WtW_LFO,WtW_MDO,WtW_BIO,
            LCV_HFO,LCV_LFO,LCV_MDO,LCV_BIO,
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
    file_name="GFI_Bunkering_Optimizer_1_1.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.caption("© 2025 — Single-vessel GFI optimizer. Initial costs shown; optimization reports deltas (selected fuel reduction & BIO increase).")
