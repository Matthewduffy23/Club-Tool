# app_rolefit_tiles.py — Role Fit Tiles (CF) with Dropdowns
# Combines: (A) Role-template matching & league-adjusted scoring + (B) rich Top-N tiles UI with per-player dropdown metrics.
# Defaults: League blend β ON at 0.40; league-mismatch penalty inside distance ON (α=1.20, p=1.50, Additive).
# Requirements: streamlit, pandas, numpy, requests, pillow, matplotlib

import io, re, math, json, time, unicodedata, uuid
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import streamlit as st
from numpy.linalg import norm
from numpy import exp

st.set_page_config(page_title="Advanced CF Scouting — Role Fit Tiles", layout="wide")
st.title("🔎 Advanced CF Scouting — Role Fit Tiles")
st.caption(
    "Role-template matching blended with league strength. Ranked by **Fit %**. "
    "Click each player to expand detailed metrics."
)

# ======================== CSV loader ========================
@st.cache_data(show_spinner=False)
def _read_csv_from_path(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)

@st.cache_data(show_spinner=False)
def _read_csv_from_bytes(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data))

def load_df(csv_name: str = "WORLDJUNE25.csv") -> pd.DataFrame:
    candidates = [Path.cwd() / csv_name, Path.cwd().parent / csv_name]
    try:
        here = Path(__file__).resolve().parent
        candidates += [here / csv_name, here.parent / csv_name]
    except Exception:
        pass
    for p in candidates:
        if p.exists():
            return _read_csv_from_path(str(p))
    st.warning(f"Could not find **{csv_name}**. Please upload below.")
    up = st.file_uploader("Upload WORLDJUNE25.csv", type=["csv"])
    if up is None:
        st.stop()
    return _read_csv_from_bytes(up.getvalue())

df = load_df("WORLDJUNE25.csv")

# ======================== leagues & strengths ========================
INCLUDED_LEAGUES = [
    'England 1.','England 2.','England 3.','England 4.','England 5.','England 6.','England 7.','England 8.','England 9.','England 10.',
    'Albania 1.','Algeria 1.','Andorra 1.','Argentina 1.','Armenia 1.','Australia 1.','Austria 1.','Austria 2.','Azerbaijan 1.','Belgium 1.',
    'Belgium 2.','Bolivia 1.','Bosnia 1.','Brazil 1.','Brazil 2.','Brazil 3.','Bulgaria 1.','Canada 1.','Chile 1.','Colombia 1.',
    'Costa Rica 1.','Croatia 1.','Cyprus 1.','Czech 1.','Czech 2.','Denmark 1.','Denmark 2.','Ecuador 1.','Egypt 1.','Estonia 1.',
    'Finland 1.','France 1.','France 2.','France 3.','Georgia 1.','Germany 1.','Germany 2.','Germany 3.','Germany 4.','Greece 1.',
    'Hungary 1.','Iceland 1.','Israel 1.','Israel 2.','Italy 1.','Italy 2.','Italy 3.','Japan 1.','Japan 2.','Kazakhstan 1.',
    'Korea 1.','Latvia 1.','Lithuania 1.','Malta 1.','Mexico 1.','Moldova 1.','Morocco 1.','Netherlands 1.','Netherlands 2.',
    'North Macedonia 1.','Northern Ireland 1.','Norway 1.','Norway 2.','Paraguay 1.','Peru 1.','Poland 1.','Poland 2.',
    'Portugal 1.','Portugal 2.','Portugal 3.','Qatar 1.','Ireland 1.','Romania 1.','Russia 1.','Saudi 1.','Scotland 1.','Scotland 2.',
    'Scotland 3.','Serbia 1.','Serbia 2.','Slovakia 1.','Slovakia 2.','Slovenia 1.','Slovenia 2.','South Africa 1.','Spain 1.','Spain 2.',
    'Spain 3.','Sweden 1.','Sweden 2.','Switzerland 1.','Switzerland 2.','Tunisia 1.','Turkey 1.','Turkey 2.','Ukraine 1.','UAE 1.',
    'USA 1.','USA 2.','Uruguay 1.','Uzbekistan 1.','Venezuela 1.','Wales 1.'
]
PRESET_LEAGUES = {
    "Top 5 Europe": {'England 1.','France 1.','Germany 1.','Italy 1.','Spain 1.'},
    "Top 20 Europe": {'England 1.','Italy 1.','Spain 1.','Germany 1.','France 1.','England 2.','Portugal 1.','Belgium 1.','Turkey 1.','Germany 2.','Spain 2.','France 2.','Netherlands 1.','Austria 1.','Switzerland 1.','Denmark 1.','Croatia 1.','Italy 2.','Czech 1.','Norway 1.'},
    "EFL (England 2–4)": {'England 2.','England 3.','England 4.'}
}
LEAGUE_STRENGTHS = {
    'England 1.':100.00,'Italy 1.':97.14,'Spain 1.':94.29,'Germany 1.':94.29,'France 1.':91.43,
    'Brazil 1.':82.86,'England 2.':71.43,'Portugal 1.':71.43,'Argentina 1.':71.43,
    'Belgium 1.':68.57,'Mexico 1.':68.57,'Turkey 1.':65.71,'Germany 2.':65.71,'Spain 2.':65.71,
    'France 2.':65.71,'USA 1.':65.71,'Russia 1.':65.71,'Colombia 1.':62.86,'Netherlands 1.':62.86,
    'Austria 1.':62.86,'Switzerland 1.':62.86,'Denmark 1.':62.86,'Croatia 1.':62.86,
    'Japan 1.':62.86,'Korea 1.':62.86,'Italy 2.':62.86,'Czech 1.':57.14,'Norway 1.':57.14,
    'Poland 1.':57.14,'Romania 1.':57.14,'Israel 1.':57.14,'Algeria 1.':57.14,'Paraguay 1.':57.14,
    'Saudi 1.':57.14,'Uruguay 1.':57.14,'Morocco 1.':57.00,'Brazil 2.':56.00,'Ukraine 1.':55.00,
    'Ecuador 1.':54.29,'Spain 3.':54.29,'Scotland 1.':58.00,'Chile 1.':51.43,'Cyprus 1.':51.43,
    'Portugal 2.':51.43,'Slovakia 1.':51.43,'Australia 1.':51.43,'Hungary 1.':51.43,'Egypt 1.':51.43,
    'England 3.':51.43,'France 3.':48.00,'Japan 2.':48.00,'Bulgaria 1.':48.57,'Slovenia 1.':48.57,
    'Venezuela 1.':48.00,'Germany 3.':45.71,'Albania 1.':44.00,'Serbia 1.':42.86,'Belgium 2.':42.86,
    'Bosnia 1.':42.86,'Kosovo 1.':42.86,'Nigeria 1.':42.86,'Azerbaijan 1.':50.00,'Bolivia 1.':50.00,
    'Costa Rica 1.':50.00,'South Africa 1.':50.00,'UAE 1.':50.00,'Georgia 1.':40.00,'Finland 1.':40.00,
    'Italy 3.':40.00,'Peru 1.':40.00,'Tunisia 1.':40.00,'USA 2.':40.00,'Armenia 1.':40.00,
    'North Macedonia 1.':40.00,'Qatar 1.':40.00,'Uzbekistan 1.':42.00,'Norway 2.':42.00,
    'Kazakhstan 1.':42.00,'Poland 2.':38.00,'Denmark 2.':37.00,'Czech 2.':37.14,'Israel 2.':37.14,
    'Netherlands 2.':37.14,'Switzerland 2.':37.14,'Iceland 1.':34.29,'Ireland 1.':34.29,'Sweden 2.':34.29,
    'Germany 4.':34.29,'Malta 1.':30.00,'Turkey 2.':31.43,'Canada 1.':28.57,'England 4.':28.57,
    'Scotland 2.':28.57,'Moldova 1.':28.57,'Austria 2.':25.71,'Lithuania 1.':25.71,'Brazil 3.':25.00,
    'England 7.':25.00,'Slovenia 2.':22.00,'Latvia 1.':22.86,'Serbia 2.':20.00,'Slovakia 2.':20.00,
    'England 9.':20.00,'England 8.':15.00,'Montenegro 1.':14.29,'Wales 1.':12.00,'Portugal 3.':11.43,
    'Northern Ireland 1.':11.43,'England 10.':10.00,'Scotland 3.':10.00,'England 6.':10.00
}

# ======================== schema & coercions ========================
ROLE_FEATURES = [
    'Touches in box per 90','xG per 90','Dribbles per 90','Progressive runs per 90',
    'Aerial duels per 90','Aerial duels won, %','Passes per 90','Non-penalty goals per 90','Accurate passes, %'
]
REQUIRED_BASE = {"Player","Team","League","Age","Position","Minutes played","Market value","Goals"}
need = set(ROLE_FEATURES) | REQUIRED_BASE
miss = [c for c in need if c not in df.columns]
if miss:
    st.error(f"Dataset missing required columns: {miss}")
    st.stop()

for c in ["Minutes played","Age","Market value","Goals"] + ROLE_FEATURES:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# ======================== sidebar: candidate pool filters ========================
with st.sidebar:
    st.header("Candidate Pool Filters (affect MATCHES only)")
    c1, c2, c3 = st.columns(3)
    use_top5  = c1.checkbox("Top-5", False)
    use_top20 = c2.checkbox("Top-20", False)
    use_efl   = c3.checkbox("EFL", False)
    seed = set()
    if use_top5:  seed |= PRESET_LEAGUES["Top 5 Europe"]
    if use_top20: seed |= PRESET_LEAGUES["Top 20 Europe"]
    if use_efl:   seed |= PRESET_LEAGUES["EFL (England 2–4)"]

    leagues_avail = sorted(set(INCLUDED_LEAGUES) | set(df["League"].dropna().unique()))
    default_leagues = sorted(seed) if seed else INCLUDED_LEAGUES
    leagues_sel = st.multiselect("Leagues in candidate pool", leagues_avail, default=default_leagues)

    pos_prefix = st.text_input("Position startswith", "CF").upper()
    position_filter = lambda p: str(p).strip().upper().startswith(pos_prefix)

    min_minutes, max_minutes = st.slider("Minutes played (pool)", 0, 6000, (1000, 6000))
    age_min_data = int(np.nanmin(df["Age"])) if df["Age"].notna().any() else 14
    age_max_data = int(np.nanmax(df["Age"])) if df["Age"].notna().any() else 50
    min_age, max_age = st.slider("Age (pool)", age_min_data, age_max_data, (16, 50))

    st.markdown("**Market value (€)**")
    mv_max_raw = int(np.nanmax(df["Market value"])) if df["Market value"].notna().any() else 50_000_000
    mv_cap = int(math.ceil(mv_max_raw / 5_000_000) * 5_000_000)
    use_m = st.checkbox("Adjust in millions", True)
    if use_m:
        max_m = max(1, mv_cap // 1_000_000)
        mv_min_m, mv_max_m = st.slider("Range (M€)", 0, max_m, (0, min(max_m, 10)))
        pool_min_value, pool_max_value = mv_min_m*1_000_000, mv_max_m*1_000_000
    else:
        pool_min_value, pool_max_value = st.slider("Range (€)", 0, mv_cap, (0, min(mv_cap, 10_000_000)), step=100_000)

    # League quality window (filter)
    min_strength, max_strength = st.slider(
        "League quality (strength)", 0, 101, (0, 101),
        help="Filter candidates by league strength (0–100)."
    )

    # Role score settings
    st.subheader("Role Score")
    decay_rate = st.slider("Exp. decay (↑=stricter)", 0.5, 10.0, 5.0, 0.5)

    # DEFAULT: β ON at 0.40
    use_league_weighting = st.checkbox("Blend in league strength (β)", value=True)
    beta = st.slider("β (0–1)", 0.0, 1.0, 0.40, 0.05, help="0=distance only, 1=league strength only")

    # League mismatch INSIDE distance — stronger by scaling with base distance spread
    use_league_mismatch = st.checkbox(
        "Penalise league mismatch inside distance (α, p)", value=True,
        help="Adds a distance-penalty based on |candidate league − template league|."
    )
    alpha = st.slider("League mismatch weight α", 0.0, 5.0, 1.20, 0.05,
                      help="Scales mismatch; combined with distance spread for real impact.")
    p_exp = st.slider("League mismatch exponent p", 1.0, 3.0, 1.50, 0.10,
                      help=">1 makes large league gaps disproportionately harsher.")
    penalty_mode = st.selectbox(
        "Penalty combine mode",
        ["Additive (stronger)", "Quadrature (gentler)"],
        index=0,
        help="Additive: base + scaled penalty (strong). Quadrature: sqrt(base^2 + penalty^2) (gentler)."
    )

    top_n = st.number_input("How many tiles", 5, 100, 20, 5)

# ======================== candidate pool (affected by sidebar) ========================
df_pool = df[df["League"].isin(leagues_sel)].copy()
df_pool = df_pool[df_pool["Position"].apply(position_filter)]
df_pool = df_pool[df_pool["Minutes played"].between(min_minutes, max_minutes)]
df_pool = df_pool[df_pool["Age"].between(min_age, max_age)]
df_pool["League Strength"] = df_pool["League"].map(LEAGUE_STRENGTHS).fillna(0.0)
df_pool = df_pool[
    (df_pool["League Strength"] >= float(min_strength)) &
    (df_pool["League Strength"] <= float(max_strength))
]
df_pool = df_pool[df_pool["Market value"].between(pool_min_value, pool_max_value)]
df_pool = df_pool.dropna(subset=ROLE_FEATURES)

if df_pool.empty:
    st.warning("No players in candidate pool after filters. Loosen filters.")
    st.stop()

# ======================== template (RAW df, unaffected by pool filters) ========================
st.markdown("---")
st.header("🎯 Club / Team Role Template")

template_league_list = sorted([str(x) for x in df["League"].dropna().unique()])
template_league = st.selectbox("Template league (scopes team list)", template_league_list)

templ_teams_all = sorted(df.loc[df["League"].astype(str) == template_league, "Team"].dropna().astype(str).unique())
search = st.text_input("Search team (filters list)", "")
templ_teams = [t for t in templ_teams_all if search.lower() in t.lower()] or templ_teams_all
template_team = st.selectbox("Template team", templ_teams)

min_minutes_template = st.slider("Minimum minutes for template CFs", 0, 6000, 1000, 100)

use_single_template_player = st.checkbox("Use single player only (else avg of team CFs)", False)

tmpl_pos_ok = lambda p: str(p).strip().upper().startswith("CF")
df_template_source = df[
    (df["League"].astype(str) == template_league) &
    (df["Team"].astype(str) == template_team) &
    (df["Position"].apply(tmpl_pos_ok)) &
    (pd.to_numeric(df["Minutes played"], errors="coerce") >= min_minutes_template)
].copy().dropna(subset=ROLE_FEATURES)

players_in_team_raw = sorted(df_template_source["Player"].dropna().astype(str).unique())
template_player_name = st.selectbox(
    "Template player (if single)",
    options=["— Select a player —"] + players_in_team_raw if players_in_team_raw else ["— none available —"],
    index=0
)

def add_role_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    f = frame.copy()
    f["Opportunities"]      = 0.7*f['Touches in box per 90'] + 0.3*f['xG per 90']
    f["Ball Carrying"]      = 0.65*f['Dribbles per 90'] + 0.35*f['Progressive runs per 90']
    f["Aerial Requirement"] = f['Aerial duels per 90'] * f['Aerial duels won, %'] / 100.0
    f["Passing Volume"]     = f['Passes per 90']
    f["Goal Output"]        = f['Non-penalty goals per 90']
    f["Retention"]          = f['Accurate passes, %']
    return f

TEMPLATE_METRICS = ["Opportunities","Ball Carrying","Aerial Requirement","Passing Volume","Goal Output","Retention"]

if df_template_source.empty:
    st.error("No CFs found for that (league, team) in RAW data (check minutes threshold or team name).")
    st.stop()

template_df = add_role_metrics(df_template_source)
if use_single_template_player:
    picked = (template_player_name if (template_player_name and not str(template_player_name).startswith("—")) else None)
    template_df = template_df[template_df["Player"].astype(str) == (picked or "__no_match__")]

if template_df.empty:
    st.error("No template player(s) after selection. Pick a player or untick single-player.")
    st.stop()

st.subheader("🧩 Players used for Role Template")
st.dataframe(
    template_df[["Player","Minutes played","Position","League"]].sort_values("Minutes played", ascending=False),
    use_container_width=True
)

template_vector = template_df[TEMPLATE_METRICS].mean()
template_strength = float(LEAGUE_STRENGTHS.get(template_league, 0.0))

# ======================== matching (on candidate pool) ========================
cf_pool = add_role_metrics(df_pool)
# exclude same team+league as template so you don't match to yourself
cf_pool = cf_pool[~((cf_pool["Team"].astype(str) == template_team) & (cf_pool["League"].astype(str) == template_league))].copy()

# hard scouting caps
cf_pool = cf_pool[(cf_pool["Age"] <= 26) & (cf_pool["Market value"] <= 10_000_000) & (cf_pool["Minutes played"] >= 1000)]
if cf_pool.empty:
    st.warning("No candidates after age/value/minutes caps. Loosen pool filters or caps.")
    st.stop()

# --- First compute BASE distance (without penalty) to know its spread ---

def base_row_dist(row):
    return norm([
        row['Opportunities']      - template_vector['Opportunities'],
        row['Ball Carrying']      - template_vector['Ball Carrying'],
        row['Aerial Requirement'] - template_vector['Aerial Requirement'],
        row['Passing Volume']     - template_vector['Passing Volume'],
        row['Goal Output']        - template_vector['Goal Output'],
        row['Retention']          - template_vector['Retention'],
    ])

cf_pool["_base_dist"] = cf_pool.apply(base_row_dist, axis=1)
base_min, base_max = float(cf_pool["_base_dist"].min()), float(cf_pool["_base_dist"].max())
base_rng = max(1e-9, base_max - base_min)  # spread used to scale penalty so it has teeth


def row_dist(row):
    base = row["_base_dist"]
    if not use_league_mismatch:
        return base

    ls_cand = float(LEAGUE_STRENGTHS.get(str(row['League']), 0.0))   # 0–100
    delta = abs(ls_cand - template_strength) / 100.0                 # 0–1
    scaled_penalty = alpha * (delta ** p_exp) * base_rng

    if penalty_mode.startswith("Additive"):
        return base + scaled_penalty
    else:
        return math.sqrt(base*base + scaled_penalty*scaled_penalty)

cf_pool["Role Fit Distance"] = cf_pool.apply(row_dist, axis=1)

# base exp-decay score
min_d = float(cf_pool["Role Fit Distance"].min()); max_d = float(cf_pool["Role Fit Distance"].max())
rng = max_d - min_d
if rng <= 1e-12:
    base_score = pd.Series(100.0, index=cf_pool.index)
else:
    base_score = 100.0 * exp(-decay_rate * ((cf_pool["Role Fit Distance"] - min_d) / rng))

# league weighting (candidate league only)
if use_league_weighting:
    league_part = cf_pool["League"].map(LEAGUE_STRENGTHS).fillna(0.0)  # 0–100 scale
else:
    league_part = 0.0

cf_pool["Fit %"] = (1.0 - beta) * base_score + beta * league_part
ranked = cf_pool.sort_values("Fit %", ascending=False).reset_index(drop=True)

st.markdown("---")
st.header("🏅 Top Role Fits — Tiles")

# ====================== Percentiles for dropdowns ======================
# We compute percentile ranks league-wise to keep context fair.
DROPDOWN_FEATURES = [
    'Defensive duels per 90','Defensive duels won, %','Aerial duels per 90','Aerial duels won, %','PAdj Interceptions',
    'Non-penalty goals per 90','xG per 90','Shots per 90','Shots on target, %','Goal conversion, %','Crosses per 90',
    'Accurate crosses, %','Dribbles per 90','Successful dribbles, %','Head goals per 90','Key passes per 90',
    'Touches in box per 90','Progressive runs per 90','Accelerations per 90','Passes per 90','Accurate passes, %',
    'xA per 90','Passes to penalty area per 90','Accurate passes to penalty area, %','Deep completions per 90','Smart passes per 90',
    'Offensive duels per 90','Offensive duels won, %'
]

# Only keep features that exist in df_pool
DROPDOWN_FEATURES = [c for c in DROPDOWN_FEATURES if c in df_pool.columns]
for feat in DROPDOWN_FEATURES:
    df_pool[feat] = pd.to_numeric(df_pool[feat], errors="coerce")
    df_pool[f"{feat} Percentile"] = df_pool.groupby("League")[feat].transform(lambda x: x.rank(pct=True) * 100.0)

# ====================== UI helpers (tiles) ======================
st.markdown("""
<style>
  :root { --bg:#0f1115; --card:#161a22; --muted:#a8b3cf; --soft:#202633; }
  .block-container { padding-top:.8rem; }
  body{ background:var(--bg); font-family: system-ui,-apple-system,'Segoe UI','Segoe UI Emoji',Roboto,Helvetica,Arial,sans-serif;}
  .wrap{ display:flex; justify-content:center; }
  .player-card{
    width:min(420px,96%); display:grid; grid-template-columns:96px 1fr 64px;
    gap:12px; align-items:start; background:var(--card); border:1px solid #252b3a;
    border-radius:18px; padding:16px;
  }
  .avatar{
    width:96px; height:96px; border-radius:12px;
    background-color:#0b0d12; background-size:cover; background-position:center;
    border:1px solid #2a3145;
  }
  .leftcol{ display:flex; flex-direction:column; align-items:center; gap:8px; }
  .name{ font-weight:800; font-size:22px; color:#e8ecff; margin-bottom:6px; }
  .sub{ color:#a8b3cf; font-size:15px; }
  .pill{ padding:2px 10px; border-radius:9px; font-weight:800; font-size:18px; color:#0b0d12; display:inline-block; min-width:42px; text-align:center; }
  .row{ display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin:4px 0; }
  .chip{ background:var(--soft); color:#cbd5f5; border:1px solid #2d3550; padding:3px 10px; border-radius:10px; font-size:13px; line-height:18px; }
  .flagchip{ display:inline-flex; align-items:center; gap:6px; background:var(--soft); color:#cbd5f5; border:1px solid #2d3550; padding:2px 8px; border-radius:10px; font-size:13px; height:22px;}
  .flagchip img{ width:18px; height:14px; border-radius:2px; display:block; }
  .pos{ color:#eaf0ff; font-weight:700; padding:4px 10px; border-radius:10px; font-size:12px; border:1px solid rgba(255,255,255,.08); }
  .teamline{ color:#e6ebff; font-size:15px; font-weight:400; margin-top:2px; }
  .rank{ color:#94a0c6; font-weight:800; font-size:18px; text-align:right; }
  .divider{ height:12px; }

  .metric-section{ background:#121621; border:1px solid #242b3b; border-radius:14px; padding:10px 12px; }
  .m-title{ color:#e8ecff; font-weight:800; letter-spacing:.02em; margin:4px 0 10px 0; font-size:20px; text-transform:uppercase; }
  .m-row{ display:flex; justify-content:space-between; align-items:center; padding:8px 8px; border-radius:10px; }
  .m-row + .m-row{ margin-top:6px; }
  .m-label{ color:#c9d3f2; font-size:16px; }
  .m-right{ display:flex; align-items:center; gap:8px; }
  .m-badge{ min-width:40px; text-align:center; padding:2px 10px; border-radius:8px; font-weight:800; font-size:18px; color:#0b0d12; border:1px solid rgba(0,0,0,.15); }
  .metrics-grid{ display:grid; grid-template-columns:1fr; gap:12px; }
  @media (min-width: 720px){ .metrics-grid{ grid-template-columns:repeat(3, 1fr); } }
</style>
""", unsafe_allow_html=True)

PALETTE=[(0,(208,2,27)),(50,(245,166,35)),(65,(248,231,28)),(75,(126,211,33)),(85,(65,117,5)),(100,(40,90,4))]

def _lerp(a,b,t): return tuple(int(round(a[i]+(b[i]-a[i])*t)) for i in range(3))

def rating_color(v: float) -> str:
    v=max(0.0,min(100.0,float(v)))
    for i in range(len(PALETTE)-1):
        x0,c0=PALETTE[i]; x1,c1=PALETTE[i+1]
        if v<=x1:
            t=0 if x1==x0 else (v-x0)/(x1-x0); r,g,b=_lerp(c0,c1,t); return f"rgb({r},{g},{b})"
    r,g,b=PALETTE[-1][1]; return f"rgb({r},{g},{b})"

POS_COLORS={
    "CF":"#183153","LWF":"#1f3f8c","LW":"#1f3f8c","LAMF":"#1f3f8c","RW":"#1f3f8c","RWF":"#1f3f8c","RAMF":"#1f3f8c",
    "AMF":"#87d37c","LCMF":"#2ecc71","RCMF":"#2ecc71","RDMF":"#0e7a3b","LDMF":"#0e7a3b",
    "LWB":"#e7d000","RWB":"#e7d000","LB":"#ff8a00","RB":"#ff8a00","RCB":"#c45a00","CB":"#c45a00","LCB":"#c45a00",
}

def chip_color(p:str)->str: return POS_COLORS.get(p.strip().upper(),"#2d3550")

# ----------------- Flags (Twemoji) -----------------
COUNTRY_TO_CC = {
    "united kingdom":"gb","great britain":"gb","northern ireland":"gb",
    "england":"eng","scotland":"sct","wales":"wls",
    "ireland":"ie","republic of ireland":"ie",
    "spain":"es","france":"fr","germany":"de","italy":"it","portugal":"pt","netherlands":"nl","belgium":"be",
    "austria":"at","switzerland":"ch","denmark":"dk","sweden":"se","norway":"no","finland":"fi","iceland":"is",
    "poland":"pl","czech republic":"cz","czechia":"cz","slovakia":"sk","slovenia":"si","croatia":"hr","serbia":"rs",
    "bosnia and herzegovina":"ba","montenegro":"me","kosovo":"xk","albania":"al","greece":"gr","hungary":"hu",
    "romania":"ro","bulgaria":"bg","russia":"ru","ukraine":"ua","georgia":"ge","kazakhstan":"kz","azerbaijan":"az",
    "armenia":"am","turkey":"tr","qatar":"qa","saudi arabia":"sa","uae":"ae","israel":"il","morocco":"ma",
    "algeria":"dz","tunisia":"tn","egypt":"eg","nigeria":"ng","ghana":"gh","senegal":"sn","ivory coast":"ci",
    "cote d'ivoire":"ci","south africa":"za","brazil":"br","argentina":"ar","uruguay":"uy","chile":"cl",
    "colombia":"co","peru":"pe","ecuador":"ec","paraguay":"py","bolivia":"bo","mexico":"mx","canada":"ca",
    "united states":"us","usa":"us","japan":"jp","korea":"kr","south korea":"kr","china":"cn","australia":"au",
    "new zealand":"nz","latvia":"lv","lithuania":"lt","estonia":"ee","moldova":"md","north macedonia":"mk",
    "malta":"mt","cyprus":"cy","luxembourg":"lu","andorra":"ad","monaco":"mc","san marino":"sm",
}
TWEMOJI_SPECIAL = {
    "eng": "1f3f4-e0067-e0062-e0065-e006e-e0067-e007f",
    "sct": "1f3f4-e0067-e0062-e0073-e0063-e006f-e0074-e007f",
    "wls": "1f3f4-e0067-e0062-e0077-e0061-e006c-e007f",
}

def country_norm(s: str) -> str:
    if not s: return ""
    return unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii").strip().lower()

def cc_to_twemoji_code(cc: str) -> str | None:
    if not cc or len(cc) != 2:
        return None
    a, b = cc.upper()
    cp1 = 0x1F1E6 + (ord(a) - ord('A'))
    cp2 = 0x1F1E6 + (ord(b) - ord('A'))
    return f"{cp1:04x}-{cp2:04x}"

def flag_img_html(country_name: str) -> str:
    n = country_norm(country_name)
    cc = COUNTRY_TO_CC.get(n, "")
    if cc in TWEMOJI_SPECIAL:
        code = TWEMOJI_SPECIAL[cc]
        src = f"https://cdnjs.cloudflare.com/ajax/libs/twemoji/14.0.2/svg/{code}.svg"
        return f"<span class='flagchip'><img src='{src}' alt='{country_name}'></span>"
    code = cc_to_twemoji_code(cc) if len(cc) == 2 else None
    if code:
        src = f"https://cdnjs.cloudflare.com/ajax/libs/twemoji/14.0.2/svg/{code}.svg"
        return f"<span class='flagchip'><img src='{src}' alt='{country_name}'></span>"
    return "<span class='chip'>—</span>"

# ====================== PlaymakerStats image resolver ======================
_PS_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.playmakerstats.com/",
}

def _http_get_text(url: str, retries: int = 1, timeout: int = 12) -> str:
    import requests
    for _ in range(retries + 1):
        try:
            r = requests.get(url, headers=_PS_HEADERS, timeout=timeout)
            if r.status_code == 200:
                return r.text
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(0.6); continue
        except Exception:
            time.sleep(0.25); continue
    return ""

def _extract_og_image(html: str) -> str | None:
    m = re.search(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']', html, flags=re.I)
    return m.group(1) if m else None

@st.cache_data(show_spinner=False)
def _pick_ps_player_link(search_html: str, surname: str, team: str) -> str | None:
    links = re.findall(r'href=["\'](/(?:player|jogador)\.php\?id=\d+[^"\']*)', search_html, flags=re.I)
    if not links:
        return None
    sn = country_norm(surname); tn = country_norm(team)
    best = None; best_score = -1
    for rel in links:
        m = re.search(re.escape(rel).replace("/", r"\/"), search_html)
        ctx = ""
        if m:
            start = max(0, m.start()-140); end = min(len(search_html), m.end()+140)
            ctx = country_norm(search_html[start:end])
        s = 0
        if sn and sn in ctx: s += 2
        if tn and tn in ctx: s += 3
        if rel.startswith("/player.php"): s += 1
        if s > best_score:
            best_score, best = s, rel
    return "https://www.playmakerstats.com" + best if best else None

@st.cache_data(show_spinner=False, ttl=24*3600)
def playmakerstats_image_by_surname_team(surname: str, team: str) -> str | None:
    from urllib.parse import quote
    q = f"{surname} {team}".strip()
    # 1) PlaymakerStats (EN)
    search_url = f"https://www.playmakerstats.com/search.php?search={quote(q)}"
    html = _http_get_text(search_url, retries=1)
    if html:
        pl_url = _pick_ps_player_link(html, surname, team)
        if pl_url:
            p_html = _http_get_text(pl_url, retries=1)
            if p_html:
                img = _extract_og_image(p_html)
                if img: return img
    # 2) zerozero.pt fallback
    search_url2 = f"https://www.zerozero.pt/procura.php?search={quote(q)}"
    html2 = _http_get_text(search_url2, retries=1)
    if html2:
        links = re.findall(r'href=["\'](/jogador\.php\?id=\d+[^"\']*)', html2, flags=re.I)
        if links:
            rel = links[0]
            pl_url2 = "https://www.zerozero.pt" + rel
            p_html2 = _http_get_text(pl_url2, retries=1)
            if p_html2:
                img2 = _extract_og_image(p_html2)
                if img2: return img2
    return None

PLACEHOLDER_IMG = "https://i.redd.it/43axcjdu59nd1.jpeg"

# Helper for reading simple fields

def get_foot_text(row: pd.Series) -> str:
    for c in ["Foot","Preferred foot","Preferred Foot"]:
        if c in row and isinstance(row[c], str) and row[c].strip():
            return row[c].strip()
    return ""

# ====================== RENDER TILES ======================
show_table = st.checkbox("Show compact table under tiles", True)

subset = ranked.head(int(top_n)).copy().reset_index(drop=True)

for idx,row in subset.iterrows():
    rank = idx+1
    surname = str(row.get("Player","")) or ""
    team    = str(row.get("Team","")) or ""
    league  = str(row.get("League","")) or ""
    pos_full= str(row.get("Position","")) or ""
    age     = int(row.get("Age",0)) if not pd.isna(row.get("Age",np.nan)) else 0
    birth_country = str(row.get("Birth country","") or "") if "Birth country" in row else ""
    foot_txt = get_foot_text(row)

    # Avatar via PlaymakerStats
    avatar_url = playmakerstats_image_by_surname_team(surname, team) or PLACEHOLDER_IMG

    fit_val = float(row.get("Fit %", 0.0))
    fit_i = min(99, int(round(fit_val)))
    ov_style = f"background:{rating_color(fit_i)};"

    # position chips (CF first)
    codes = [c for c in re.split(r"[,/; ]+", pos_full.strip().upper()) if c]
    if "CF" in codes: codes = ["CF"] + [c for c in codes if c!="CF"]
    chips_html = "".join(f"<span class='pos' style='background:{chip_color(c)}'>{c}</span> " for c in dict.fromkeys(codes))

    # flag + meta
    def flag_and_meta_html(country_name: str, age: int, foot_txt: str) -> str:
        flag_html = flag_img_html(country_name) if country_name else ""
        age_chip = f"<span class='chip'>{age}y.o.</span>" if age>0 else ""
        foot_row = f"<div class='row'><span class='chip'>{foot_txt}</span></div>" if foot_txt else "<div class='row'></div>"
        top_row = f"<div class='row'>{flag_html}{age_chip}</div>"
        return top_row + foot_row

    st.markdown(f"""
    <div class='wrap'>
      <div class='player-card'>
        <div class='leftcol'>
          <div class='avatar' style="background-image:url('{avatar_url}');"></div>
          {flag_and_meta_html(birth_country, age, foot_txt)}
        </div>
        <div>
          <div class='name'>{surname}</div>
          <div class='row' style='align-items:center;'>
            <span class='pill' style='{ov_style}'>{fit_i}</span>
            <span class='sub'>Fit %</span>
          </div>
          <div class='row'>{chips_html}</div>
          <div class='teamline'>{team} · {league}</div>
        </div>
        <div class='rank'>#{rank}</div>
      </div>
    </div>
    <div class='divider'></div>
    """, unsafe_allow_html=True)

    # === dropdown with individual metrics ===
    with st.expander("▼ Show individual metrics"):
        def pct_of_row(row: pd.Series, metric: str) -> float:
            col = f"{metric} Percentile"
            v = float(row[col]) if col in df_pool.columns and col in row and not pd.isna(row[col]) else np.nan
            if np.isnan(v):
                # fallback: compute percentile within current league for this one metric on the fly
                try:
                    vals = pd.to_numeric(df_pool[df_pool["League"]==row["League"]][metric], errors="coerce").dropna()
                    if len(vals):
                        v0 = float(row.get(metric, np.nan))
                        v = float((vals <= v0).mean()*100.0) if not np.isnan(v0) else 0.0
                    else:
                        v = 0.0
                except Exception:
                    v = 0.0
            return max(0.0, min(100.0, float(v)))

        def show99(x: float) -> int:
            try:
                return min(99, int(round(float(x))))
            except Exception:
                return 0

        def metrics_section_html(title: str, items: list[tuple[str, float, str]]) -> str:
            rows = []
            for lab, pct, val_str in items:
                pct_i = show99(pct)
                rows.append(
                    f"<div class='m-row'>"
                    f"  <div class='m-label'>{lab}</div>"
                    f"  <div class='m-right'><span class='m-badge' style='background:{rating_color(pct_i)}'>{pct_i}</span></div>"
                    f"</div>"
                )
            return f"<div class='metric-section'><div class='m-title'>{title}</div>{''.join(rows)}</div>"

        def val_str(row: pd.Series, metric: str) -> str:
            v = row.get(metric, None)
            if v is None or (isinstance(v, float) and np.isnan(v)): return "—"
            return (f"{float(v):.0f}%" if "%" in metric else f"{float(v):.2f}") if isinstance(v,(int,float,np.floating)) else str(v)

        ATTACKING = []
        for lab, met in [
            ("Goals: Non-Penalty","Non-penalty goals per 90"),
            ("xG","xG per 90"),
            ("Shots","Shots per 90"),
            ("Header Goals","Head goals per 90"),
            ("Expected Assists","xA per 90"),
            ("Progressive Runs","Progressive runs per 90"),
            ("Touches in Opposition Box","Touches in box per 90"),
        ]:
            if met in df_pool.columns:
                ATTACKING.append((lab, pct_of_row(row, met), val_str(row, met)))

        DEFENSIVE = []
        for lab, met in [
            ("Aerial Duels","Aerial duels per 90"),
            ("Aerial Duel Success %","Aerial duels won, %"),
            ("PAdj. Interceptions","PAdj Interceptions"),
            ("Defensive Duels","Defensive duels per 90"),
            ("Defensive Duel Success %","Defensive duels won, %"),
        ]:
            if met in df_pool.columns:
                DEFENSIVE.append((lab, pct_of_row(row, met), val_str(row, met)))

        POSSESSION = []
        for lab, met in [
            ("Dribbles","Dribbles per 90"),
            ("Dribbling Success %","Successful dribbles, %"),
            ("Key Passes","Key passes per 90"),
            ("Passes","Passes per 90"),
            ("Passing Accuracy %","Accurate passes, %"),
            ("Passes to Penalty Area","Passes to penalty area per 90"),
            ("Passes to Penalty Area %","Accurate passes to penalty area, %"),
            ("Deep Completions","Deep completions per 90"),
            ("Smart Passes","Smart passes per 90"),
        ]:
            if met in df_pool.columns:
                POSSESSION.append((lab, pct_of_row(row, met), val_str(row, met)))

        col_html = (
            "<div class='metrics-grid'>"
            f"{metrics_section_html('ATTACKING', ATTACKING)}"
            f"{metrics_section_html('DEFENSIVE', DEFENSIVE)}"
            f"{metrics_section_html('POSSESSION', POSSESSION)}"
            "</div>"
        )
        st.markdown(col_html, unsafe_allow_html=True)

# ====================== Compact Table ======================
if show_table:
    st.markdown("---")
    st.subheader("📋 Top Role Fits — Table (sorted by Fit %)")
    st.dataframe(
        subset[["Player","Team","League","Age","Minutes played","Market value","Fit %"]],
        use_container_width=True
    )

# ====================== Download CSV ======================
with st.sidebar:
    st.markdown("---")
    st.download_button(
        "⬇️ Download Top-N (CSV)",
        data=subset.to_csv(index=False).encode('utf-8'),
        file_name=f"role_fit_tiles_top{int(top_n)}.csv",
        mime="text/csv",
        key=f"download_topn_csv_{uuid.uuid4().hex}"
    )



