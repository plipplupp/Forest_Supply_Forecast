"""Streamlit-dashboard: "Kan en användare förstå forecasten på 30 sekunder?"

Kör:  streamlit run app/streamlit_app.py

Visar historik + prognos + osäkerhetsintervall per region/modell/horisont,
modellens fel mot baseline samt senaste observerade värde och förväntad
förändring. All underliggande data är offentlig proxy-data (avverkningsanmälan
är INTE samma sak som faktisk avverkning).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import LANDSDEL_BY_LAN, RESULTS_DIR  # noqa: E402
from src.data.datasets import (  # noqa: E402
    load_target_wide,
    load_volume_factors,
)
from src.data.felling_volume import factor_for_region  # noqa: E402

st.set_page_config(page_title="Forest Supply Forecast", page_icon="🌲", layout="wide")

MODEL_ORDER = ["TimesFM (univariat)", "XGBoost", "Seasonal Naive"]
MODEL_KEY = {"TimesFM (univariat)": "timesfm_univ", "XGBoost": "xgb_base",
             "Seasonal Naive": "seasonal_naive"}
MODEL_INFO = {
    "TimesFM (univariat)": (
        "Univariat: prognosen bygger ENDAST på regionens egen historik, inga andra "
        "indata. TimesFM 3 är en av Google förtränad ”foundation model” för tidsserier "
        "(zero-shot – aldrig tränad på svensk skogsdata)."
    ),
    "XGBoost": (
        "Klassisk maskininlärning: tränas per region på förskjutna värden "
        "(lag-värden, 1/2/3/6/12 mån), rullande medel/std och kalender – omtränad vid "
        "varje prognostillfälle med enbart den historik som fanns då."
    ),
    "Seasonal Naive": (
        "Enkel baseline: prognosen för en månad är samma utfall som samma månad "
        "förra året. Visar vad man får ”gratis” av årscykeln – modellerna måste "
        "slå detta."
    ),
}


def sv_num(x: float, decimals: int = 0) -> str:
    """Svenskt talformat: mellanslag som tusentalsavgränsare, komma som decimaltecken."""
    s = f"{x:,.{decimals}f}"
    return s.replace(",", " ").replace(".", ",")


@st.cache_data
def load_data():
    wide = load_target_wide()
    fc = pd.read_parquet(RESULTS_DIR / "latest_forecast.parquet")
    metrics = pd.read_csv(RESULTS_DIR / "metrics_per_region.csv")
    return wide, fc, metrics


wide, forecasts, metrics = load_data()
factors = load_volume_factors()

_top = ["Hela landet", "Norra Norrland", "Södra Norrland", "Svealand", "Götaland"]
regions = _top + sorted(r for r in wide.columns if r not in _top)
region = st.sidebar.selectbox("Region", regions,
                              index=regions.index("Götaland"))
avail = [m for m in MODEL_ORDER
         if m in set(forecasts.loc[forecasts["region_name"] == region, "model"])]
model = st.sidebar.selectbox("Modell", avail, index=0)
horizon = st.sidebar.select_slider("Prognoshorisont (månader)", options=[1, 3, 6], value=3)
visa_volym = st.sidebar.checkbox("Visa som indikativ virkesvolym (m³sk)", value=False)
st.sidebar.caption(MODEL_INFO.get(model, ""))
st.sidebar.caption(
    "Horisonterna 1, 3 och 6 är de som utvärderats i backtesting. Övriga "
    "TimesFM-varianter (kalender/väder/priser, multivariat) testades men gav ingen "
    "märkbar skillnad i precision – se docs/experiments_results.md."
)

# ------------------------------------------------------------- volymomvandling
factor, used_landsdel = factor_for_region(region, factors, LANDSDEL_BY_LAN)
scale = factor / 1000.0 if visa_volym else 1.0  # tusen m3sk per ha-visning
enhet = "tusen m³sk" if visa_volym else "ha"

# ------------------------------------------------------------- nyckeltal
series = wide[region].dropna()
last_ts = series.index[-1]
last_val = series.iloc[-1] * scale
sel = forecasts[(forecasts["region_name"] == region) & (forecasts["model"] == model)]
fcst_ha = sel.sort_values("horizon").copy()          # oskalad (ha) – för kvoter
fcst = fcst_ha.copy()
if visa_volym:
    for col in ("prediction", "p10", "p90"):
        fcst[col] = fcst[col] * scale
row = fcst[fcst["horizon"] == horizon].iloc[0]
row_ha = fcst_ha[fcst_ha["horizon"] == horizon].iloc[0]
chg = (row["prediction"] / last_val - 1) * 100
# jämför prognosmånaden mot SAMMA månad i fjol (observerat värde), rätt enhet via kvot i ha
target_prev_ts = row["timestamp"] - pd.DateOffset(years=1)
prev_same = wide[region].get(target_prev_ts, np.nan)
chg_y = (row_ha["prediction"] / prev_same - 1) * 100 if pd.notna(prev_same) else np.nan

# modellens precision vid vald horizont + baseline
mrow = metrics[(metrics["region_name"] == region)
               & (metrics["model"] == MODEL_KEY[model])
               & (metrics["horizon"] == horizon)]
mae = float(mrow["MAE"].iloc[0]) if len(mrow) else float("nan")
mase = float(mrow["MASE"].iloc[0]) if len(mrow) else float("nan")
smape = float(mrow["sMAPE"].iloc[0]) if len(mrow) else float("nan")
brow = metrics[(metrics["region_name"] == region)
               & (metrics["model"] == "seasonal_naive")
               & (metrics["horizon"] == horizon)]
base_mae = float(brow["MAE"].iloc[0]) if len(brow) else float("nan")
vs_base = (1 - mae / base_mae) * 100 if base_mae else float("nan")
typical = float(series.iloc[-36:].median()) if len(series) >= 36 else float(series.median())
mae_pct = mae / typical * 100 if typical else float("nan")

st.title("🌲 Forest Supply Forecast")
st.markdown(
    "**TimesFM 3 (zero-shot foundation model) ger i genomsnitt lägst fel över alla 26 "
    "regioner** – utvärderat med tillförlitlig rolling-origin-backtesting (48 olika "
    "startpunkter × horisonterna 1/3/6 mån), där modellen aldrig får se data från "
    "framtiden. På enskilda serier, särskilt riket/landsdelar "
    "vid längre horisonter, kan enkelbaselines vara jämbördiga eller bättre."
)
st.caption(
    "Prognos för anmäld avverkningsareal (ha) – en ledande indikator för kommande "
    "råvarutillförsel: anmälan föregår avverkning, som blir virkesvolym. Prognosen gäller "
    "arealen, inte kubikmeter."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Senast observerat ({last_ts:%Y-%m}), {enhet}", sv_num(last_val))
c2.metric(f"Prognos +{horizon} mån ({row['timestamp']:%Y-%m}), {enhet}",
          sv_num(row["prediction"]),
          f"{chg:+.0f} % mot {last_ts:%Y-%m}", delta_color="off",
          help=f"Jämförelse med senaste observerade månad ({last_ts:%Y-%m}, {sv_num(last_val)} {enhet}).")
c3.metric(f"Prognos {row['timestamp']:%Y-%m} mot {target_prev_ts:%Y-%m} (fjolåret)",
          f"{chg_y:+.0f} %" if pd.notna(chg_y) else "–",
          delta_color="off",
          help=f"Prognosen för {row['timestamp']:%Y-%m} jämförd med det observerade "
               f"utfallet samma månad föregående år ({target_prev_ts:%Y-%m}: "
               f"{sv_num(prev_same * scale)} {enhet}).")
c4.metric(f"Medelabsolut fel vid {horizon} mån (MAE)", f"{sv_num(mae)} ha",
          help="Medelabsolut fel i rolling-origin-backtesting 2022-09–2026-08 "
               "(felmåttet är alltid i hektar, även i volymvyn): "
               f"≈ {mae_pct:.0f} % av en typisk månadsnivå.")
if model == "Seasonal Naive":
    st.info("Seasonal Naive är här baselinen – övriga modeller jämförs med den. "
            f"Medelfel: {sv_num(mae)} ha vid {horizon} mån.", icon="ℹ️")
elif vs_base >= 0:
    st.success(f"{model}: {vs_base:.0f} % lägre medelfel än baseline vid {horizon} mån "
               f"(Seasonal Naive: {sv_num(base_mae)} ha) – bättre precision.")
else:
    st.error(f"{model}: {abs(vs_base):.0f} % högre medelfel än baseline vid {horizon} mån "
             f"(Seasonal Naive: {sv_num(base_mae)} ha) – sämre precision.")

# ------------------------------------------------------------- diagram
hist = wide[region].iloc[-48:].dropna() * scale
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=hist.index, y=hist.values, name="Observerat",
    line=dict(color="#2C6E8F", width=2),
    customdata=[sv_num(v) for v in hist.values],
    hovertemplate="%{x|%Y-%m}<br>Observerat: %{customdata} " + enhet + "<extra></extra>",
))
if fcst[["p10", "p90"]].notna().all(axis=None):
    fig.add_trace(go.Scatter(x=fcst["timestamp"], y=fcst["p90"], mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fcst["timestamp"], y=fcst["p10"], mode="lines",
                             fill="tonexty", fillcolor="rgba(196,106,43,0.18)",
                             line=dict(width=0), name="P10–P90 (80 %-intervall)",
                             hoverinfo="skip"))
fig.add_trace(go.Scatter(
    x=fcst["timestamp"], y=fcst["prediction"], name="Prognos",
    line=dict(color="#C46A2B", width=2, dash="dot"),
    mode="lines+markers", marker=dict(size=6),
    customdata=[sv_num(v) for v in fcst["prediction"]],
    hovertemplate="%{x|%Y-%m}<br>Prognos: %{customdata} " + ("tusen m³sk" if visa_volym else "ha") + "<extra></extra>",
))
fig.update_xaxes(hoverformat="%Y-%m")
fig.update_layout(
    height=430, margin=dict(l=10, r=10, t=10, b=10),
    yaxis_title=f"{enhet}/månad",
    legend=dict(orientation="h", y=1.02, yanchor="bottom", title=""),
    hovermode="x unified",
)
if visa_volym:
    st.markdown(f"**Indikativ virkesvolym – {region} (tusen m³sk/månad)**")
    via = ("" if used_landsdel == region else
           f" ({region} använder {used_landsdel}s omvandlingsfaktor eftersom regionen "
           f"saknar egen)")
    st.caption(
        f"Omvandling: anmäld areal × {sv_num(factor, 1)} m³sk/ha – slutavverkning, "
        f"{factors.loc[factors['landsdel'] == used_landsdel, 'period'].iloc[0]}, "
        f"Skogsstyrelsen JO0312_06{via}. Indikativt värde – verklig volym varierar med "
        "trädslag, skogsägare (ägarkategori) och avverkningstyp, och all anmäld areal "
        "realiseras inte som avverkning."
    )
else:
    st.markdown(f"**Anmäld avverkningsareal – {region} (ha/månad)**")
if "XGBoost" in model:
    st.caption("XGBoost ger punktprognoser utan prognosintervall – kvantiler "
               "(P10–P90) produceras endast av TimesFM i detta projekt.")
st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------- precision + förklaringar
with st.expander("Modellens precision – och vad felmåtten betyder"):
    bättre = "bättre" if vs_base >= 0 else "sämre"
    jam = ("Här är modellen själv baselinen – övriga modeller mäts mot den."
           if model == "Seasonal Naive" else
           f"Det är **{abs(vs_base):.0f} % {bättre}** än Seasonal Naive-baseline "
           f"(MAE {sv_num(base_mae)} ha), som bara upprepar fjolåret.")
    st.markdown(
        f"**Är medelfelet bra eller dåligt?** {model} missar i genomsnitt "
        f"**{sv_num(mae)} ha** vid {horizon} månaders horisont – ungefär "
        f"**{mae_pct:.0f} % av en typisk månad** ({sv_num(typical)} ha) i {region}. "
        + jam
    )
    st.markdown("---")
    st.markdown(
        f"**{model}** vid {horizon} månader: MAE {sv_num(mae)} ha · "
        f"MASE {mase:.2f} · sMAPE {smape:.1f} %"
    )
    st.markdown(
        "- **MAE** (Mean Absolute Error, medelabsolut fel) – i genomsnitt missar "
        "prognosen med så många hektar. Lättast att relatera till.\n"
        f"- **MASE** – felet skalat mot regeln ”samma månad som i fjol”. "
        + (f"**MASE < 1 betyder bättre än den regeln** (här: {mase:.2f}, dvs. "
           f"{(1 - mase) * 100:.0f} % mindre fel)."
           if mase < 1 else
           f"**MASE ≥ 1 betyder sämre än eller lika bra som regeln** (här: {mase:.2f}, "
           f"dvs. {(mase - 1) * 100:.0f} % mer fel).")
        + " Gör serier av olika storlek jämförbara.\n"
        "- **sMAPE** (symmetric Mean Absolute Percentage Error) – procentuellt fel där "
        "över- och underprognoser väger lika. Ett komplement till MAE när man vill ha "
        "ett procentmått som är robust även för små serier."
    )
    if "TimesFM" in model:
        st.caption("TimesFM-vikter är icke-kommersiella (forskning/demonstration).")

st.caption(
    "P10–P90 är ett 80-procentigt prognosintervall (prognosfördelning, inte garanti): "
    "i backtesting hamnade utfallet i bandet vid 79 % av prognostillfällena vid 1 mån, "
    "ca 60 % vid 6 mån."
)
