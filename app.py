# app.py — CF Role Template Scouting + League→Team template flow
# Defaults: (1) League blend β ON at 0.40, (2) Use single template player = OFF
# Stronger league-quality influence inside distance via spread-scaled additive penalty (toggleable)

import io, math, uuid
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from numpy.linalg import norm
from numpy import exp

st.set_page_config(page_title="Club Striker Scouting System", layout="wide")
st.title("🔎 Advanced Club ST Scouting System")
st.caption(
    "Club Selection "
    "Individual player specific or generalized role"
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
FEATURES = [
    'Touches in box per 90','xG per 90','Dribbles per 90','Progressive runs per 90',
    'Aerial duels per 90','Aerial duels won, %','Passes per 90','Non-penalty goals per 90','Accurate passes, %'
]
REQUIRED_BASE = {"Player","Team","League","Age","Position","Minutes played","Market value","Goals"}
need = set(FEATURES) | REQUIRED_BASE
miss = [c for c in need if c not in df.columns]
if miss:
    st.error(f"Dataset missing required columns: {miss}")
    st.stop()

for c in ["Minutes played","Age","Market value","Goals"] + FEATURES:
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
    # Wider max and higher default so it *matters*
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

    top_n = st.number_input("Top N (table)", 5, 200, 50, 5)

# ======================== candidate pool (affected by sidebar) ========================
df_pool = df[df["League"].isin(leagues_sel)].copy()
df_pool = df_pool[df_pool["Position"].apply(position_filter)]
df_pool = df_pool[df_pool["Minutes played"].between(min_minutes, max_minutes)]
df_pool = df_pool[df_pool["Age"].between(min_age, max_age)]
df_pool = df_pool[df_pool["Market value"].between(pool_min_value, pool_max_value)]
df_pool["League Strength"] = df_pool["League"].map(LEAGUE_STRENGTHS).fillna(0.0)
df_pool = df_pool[
    (df_pool["League Strength"] >= float(min_strength)) &
    (df_pool["League Strength"] <= float(max_strength))
]
df_pool = df_pool.dropna(subset=FEATURES)

if df_pool.empty:
    st.warning("No players in candidate pool after filters. Loosen filters.")
    st.stop()

# ======================== template (RAW df, unaffected by pool filters) ========================
st.markdown("---")
st.header("🎯 Club Selection")

template_league_list = sorted([str(x) for x in df["League"].dropna().unique()])
template_league = st.selectbox("Template league (scopes team list)", template_league_list)

templ_teams_all = sorted(df.loc[df["League"].astype(str) == template_league, "Team"].dropna().astype(str).unique())
search = st.text_input("Search team (filters list)", "")
templ_teams = [t for t in templ_teams_all if search.lower() in t.lower()] or templ_teams_all
template_team = st.selectbox("Template team", templ_teams)

min_minutes_template = st.slider("Minimum minutes for template CFs", 0, 6000, 1000, 100)

# DEFAULT: Use single template player OFF
use_single_template_player = st.checkbox("Use single player only (else avg of team CFs)", False)

tmpl_pos_ok = lambda p: str(p).strip().upper().startswith("CF")
df_template_source = df[
    (df["League"].astype(str) == template_league) &
    (df["Team"].astype(str) == template_team) &
    (df["Position"].apply(tmpl_pos_ok)) &
    (pd.to_numeric(df["Minutes played"], errors="coerce") >= min_minutes_template)
].copy().dropna(subset=FEATURES)

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
    # scale penalty by the spread of base distances so it matters across datasets
    scaled_penalty = alpha * (delta ** p_exp) * base_rng

    if penalty_mode.startswith("Additive"):
        return base + scaled_penalty                 # STRONGER
    else:
        return math.sqrt(base*base + scaled_penalty*scaled_penalty)  # gentler

cf_pool["Role Fit Distance"] = cf_pool.apply(row_dist, axis=1)

# base exp-decay score
min_d = float(cf_pool["Role Fit Distance"].min()); max_d = float(cf_pool["Role Fit Distance"].max())
rng = max_d - min_d
if rng <= 1e-12:
    base_score = pd.Series(100.0, index=cf_pool.index)
else:
    base_score = 100.0 * exp(-decay_rate * ((cf_pool["Role Fit Distance"] - min_d) / rng))

# league weighting (candidate league only) — optional blend (DEFAULT ON @ β=0.4)
if use_league_weighting:
    league_part = cf_pool["League"].map(LEAGUE_STRENGTHS).fillna(0.0)  # 0–100 scale
else:
    league_part = 0.0

cf_pool["Role Fit Score"] = (1.0 - beta) * base_score + beta * league_part
ranked = cf_pool.sort_values("Role Fit Score", ascending=False).reset_index(drop=True)

st.markdown("---")
st.header("🏅 Top Role Matches")
st.dataframe(
    ranked[["Player","Team","League","Age","Minutes played","Market value","Role Fit Score"]].head(int(top_n)),
    use_container_width=True
)

# ======================== Feature Z (under the table) ========================
st.markdown("---")
st.header("Advanced Individual Player Analysis")

# lazy imports so the app paints fast
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
from matplotlib import font_manager as fm
from matplotlib.font_manager import FontProperties
from PIL import Image

# Player picker (defaults to best match)
left, right = st.columns([2,2])
with left:
    options_ranked = ranked["Player"].astype(str).head(int(top_n)).tolist()
    any_pool = st.checkbox("Pick from entire candidate pool (not just Top N)", value=False)
    options = sorted(df_pool["Player"].dropna().astype(str).unique()) if any_pool else options_ranked
    if not options:
        st.info("No players available for Feature Z. Adjust filters.")
        st.stop()
    player_sel = st.selectbox("Choose player for Feature Z", options, index=0)

with right:
    show_height = st.checkbox("Show height in info row", value=True)
    foot_override_on = st.checkbox("Edit foot value", value=False)
    foot_override_text = st.text_input("Foot (e.g., Left)", value="", disabled=not foot_override_on)
    name_override_on = st.checkbox("Edit display name", value=False)
    name_override = st.text_input("Display name", "", disabled=not name_override_on)
    footer_caption_text = st.text_input("Footer caption", "Percentile Rank")

# Build row for chosen player (from candidate pool so percentiles match context)
player_row = df_pool[df_pool["Player"].astype(str) == str(player_sel)].head(1)
if player_row.empty:
    st.info("Pick a player above.")
    st.stop()

# ---------- helpers ----------
def _safe_get(sr, key, default="—"):
    try:
        v = sr.iloc[0].get(key, default)
        s = "" if v is None else str(v)
        return default if s.strip() == "" else s
    except Exception:
        return default

def pct_series(col: str) -> float:
    vals = pd.to_numeric(df_pool[col], errors="coerce").dropna()
    if vals.empty: return np.nan
    v = pd.to_numeric(player_row.iloc[0][col], errors="coerce")
    if pd.isna(v): return np.nan
    return float((vals <= v).mean() * 100.0)

def val_of(col: str):
    v = player_row.iloc[0].get(col)
    if pd.isna(v): return (np.nan, "—")
    if isinstance(v, (int,float,np.floating)):
        return (float(v), f"{float(v):.0f}%" if "%" in col else f"{float(v):.2f}")
    return (v, str(v))

# ---------- info row data ----------
pos   = _safe_get(player_row, "Position", "CF")
name_ = _safe_get(player_row, "Player", _safe_get(player_row, "Name", ""))
if name_override_on and name_override.strip():
    name_ = name_override.strip()
team  = _safe_get(player_row, "Team", "")
age_raw = _safe_get(player_row, "Age", "")
try: age = f"{float(age_raw):.0f}"
except Exception: age = age_raw
games   = _safe_get(player_row, "Matches played", _safe_get(player_row, "Games", _safe_get(player_row, "Apps", "—")))
minutes = _safe_get(player_row, "Minutes", _safe_get(player_row, "Minutes played", "—"))
goals   = _safe_get(player_row, "Goals", "—")
assists = _safe_get(player_row, "Assists", "—")
foot    = _safe_get(player_row, "Foot", _safe_get(player_row, "Preferred Foot", "—"))
foot_display = (foot_override_text.strip() if (foot_override_on and foot_override_text and foot_override_text.strip()) else foot)
height_text = ""
for col in ["Height","Height (ft)","Height ft","Height (cm)"]:
    v = _safe_get(player_row, col, "")
    if v and v != "—":
        height_text = str(v).strip()
        break

# ---------- sections ----------
ATTACKING, DEFENSIVE, POSSESSION = [], [], []
for lab, met in [
    ("Goals: Non-Penalty","Non-penalty goals per 90"),
    ("xG","xG per 90"),
    ("Shots","Shots per 90"),
    ("Header Goals","Head goals per 90"),
    ("Expected Assists","xA per 90"),
    ("Progressive Runs","Progressive runs per 90"),
    ("Touches in Opp. Box","Touches in box per 90"),
]:
    if met in df_pool.columns:
        ATTACKING.append((lab, float(np.nan_to_num(pct_series(met), nan=0.0)), val_of(met)[1]))

for lab, met in [
    ("Aerial Duels","Aerial duels per 90"),
    ("Aerial Duel Success %","Aerial duels won, %"),
    ("PAdj. Interceptions","PAdj Interceptions"),
    ("Defensive Duels","Defensive duels per 90"),
    ("Defensive Duel Success %","Defensive duels won, %"),
]:
    if met in df_pool.columns:
        DEFENSIVE.append((lab, float(np.nan_to_num(pct_series(met), nan=0.0)), val_of(met)[1]))

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
        POSSESSION.append((lab, float(np.nan_to_num(pct_series(met), nan=0.0)), val_of(met)[1]))

sections = [("Attacking",ATTACKING),("Defensive",DEFENSIVE),("Possession",POSSESSION)]
sections = [(t,lst) for t,lst in sections if lst]

# ---------- drawing ----------
def _font_name_or_fallback(pref, fallback="DejaVu Sans"):
    from matplotlib import font_manager as fm
    installed = {f.name for f in fm.fontManager.ttflist}
    for n in pref:
        if n in installed: return n
    return fallback

from matplotlib.font_manager import FontProperties
FONT_TITLE_FAMILY = _font_name_or_fallback(["Tableau Bold","Tableau Sans Bold","Tableau"])
FONT_BOOK_FAMILY  = _font_name_or_fallback(["Tableau Book","Tableau Sans","Tableau"])
TITLE_FP     = FontProperties(family=FONT_TITLE_FAMILY, weight='bold',     size=24)
H2_FP        = FontProperties(family=FONT_TITLE_FAMILY, weight='semibold', size=20)
LABEL_FP     = FontProperties(family=FONT_BOOK_FAMILY,  weight='medium',   size=10)
INFO_LABEL_FP= FontProperties(family=FONT_BOOK_FAMILY,  weight='bold',     size=10)
INFO_VALUE_FP= FontProperties(family=FONT_BOOK_FAMILY,  weight='regular',  size=10)
BAR_VALUE_FP = FontProperties(family=FONT_BOOK_FAMILY,  weight='regular',  size=8)
TICK_FP      = FontProperties(family=FONT_BOOK_FAMILY,  weight='medium',   size=10)
FOOTER_FP    = FontProperties(family=FONT_BOOK_FAMILY,  weight='medium',   size=10)

PAGE_BG = "#ebebeb"; AX_BG = "#f3f3f3"; TRACK="#d6d6d6"
TITLE_C="#111111"; LABEL_C="#222222"; DIVIDER="#000000"
ticks = np.arange(0,101,10)
LEFT, RIGHT, TOP, BOT = 0.055, 0.030, 0.035, 0.07
header_h, GAP = 0.045, 0.020
gutter = 0.215
BAR_FRAC = 0.92

def pct_to_rgb(v):
    v=float(np.clip(v,0,100))
    TAB_RED=np.array([199,54,60]); TAB_GOLD=np.array([240,197,106]); TAB_GREEN=np.array([61,166,91])
    def _blend(c1,c2,t): c=c1+(c2-c1)*np.clip(t,0,1); return f"#{int(c[0]):02x}{int(c[1]):02x}{int(c[2]):02x}"
    return _blend(TAB_RED,TAB_GOLD,v/50) if v<=50 else _blend(TAB_GOLD,TAB_GREEN,(v-50)/50)

import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation

fig_size   = (11.8, 9.6); dpi = 120
title_row_h = 0.125; header_block_h = title_row_h + 0.055
fig = plt.figure(figsize=fig_size, dpi=dpi); fig.patch.set_facecolor(PAGE_BG)

# Title
fig.text(LEFT, 1 - TOP - 0.010, f"{name_}\u2009|\u2009{team}",
         ha="left", va="top", color=TITLE_C, fontproperties=TITLE_FP)

# Info rows
def draw_pairs_line(pairs_line, y):
    x = LEFT; renderer = fig.canvas.get_renderer()
    for i,(lab,val) in enumerate(pairs_line):
        t1 = fig.text(x, y, lab, ha="left", va="top", color=LABEL_C, fontproperties=INFO_LABEL_FP)
        fig.canvas.draw(); x += t1.get_window_extent(renderer).width / fig.bbox.width
        t2 = fig.text(x, y, str(val), ha="left", va="top", color=LABEL_C, fontproperties=INFO_VALUE_FP)
        fig.canvas.draw(); x += t2.get_window_extent(renderer).width / fig.bbox.width
        if i != len(pairs_line)-1:
            t3 = fig.text(x, y, "  |  ", ha="left", va="top", color="#555555", fontproperties=INFO_VALUE_FP)
            fig.canvas.draw(); x += t3.get_window_extent(renderer).width / fig.bbox.width

row1 = [("Position: ",pos), ("Age: ",age), ("Height: ", height_text if (show_height and height_text) else "—")]
row2 = [("Games: ",games), ("Goals: ",goals), ("Assists: ",assists)]
row3 = [("Minutes: ",minutes), ("Foot: ",foot_display)]
title_y = 1 - TOP - 0.010
y1 = title_y - 0.055; y2 = y1 - 0.039; y3 = y2 - 0.039
draw_pairs_line(row1, y1); draw_pairs_line(row2, y2); draw_pairs_line(row3, y3)

# Divider
fig.lines.append(plt.Line2D([LEFT, 1 - RIGHT],[1 - TOP - header_block_h + 0.004]*2,
                            transform=fig.transFigure, color=DIVIDER, lw=0.8, alpha=0.35))

# Panels
def draw_panel(panel_top, title, tuples, *, show_xticks=False, draw_bottom_divider=True):
    n = len(tuples)
    if n == 0: return panel_top
    total_rows = sum(len(lst) for _, lst in sections)
    rows_space_total = 1 - (TOP + BOT) - header_block_h - header_h*len(sections) - GAP*(len(sections)-1)
    row_slot = rows_space_total / max(total_rows,1)

    fig.text(LEFT, panel_top - 0.012, title, ha="left", va="top", color=TITLE_C, fontproperties=H2_FP)

    ax = fig.add_axes([LEFT + gutter, panel_top - header_h - n*row_slot, 1 - LEFT - RIGHT - gutter, n*row_slot])
    ax.set_facecolor(AX_BG); ax.set_xlim(0,100); ax.set_ylim(-0.5,n-0.5)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(axis="x", bottom=False, labelbottom=False, length=0)
    ax.tick_params(axis="y", left=False,  labelleft=False,  length=0)
    ax.set_yticks([]); ax.get_yaxis().set_visible(False)

    for i in range(n):
        ax.add_patch(plt.Rectangle((0, i-(BAR_FRAC/2)), 100, BAR_FRAC, color=TRACK, ec="none", zorder=0.5))
    for gx in ticks:
        ax.vlines(gx, -0.5, n-0.5, colors=(0,0,0,0.16), linewidth=0.8, zorder=0.75)

    for i,(lab,pct,val_str) in enumerate(tuples[::-1]):
        y = i; bar_w = float(np.clip(pct,0,100))
        ax.add_patch(plt.Rectangle((0, y-(BAR_FRAC/2)), bar_w, BAR_FRAC, color=pct_to_rgb(bar_w), ec="none", zorder=1.0))
        x_text = 1.0 if bar_w >= 3 else min(100.0, bar_w + 0.8)
        ax.text(x_text, y, val_str, ha="left", va="center", color="#0B0B0B", fontproperties=BAR_VALUE_FP, zorder=2.0, clip_on=False)

    ax.axvline(50, color="#000000", ls=(0,(4,4)), lw=1.5, alpha=0.7, zorder=3.5)

    for i,(lab,_,_) in enumerate(tuples[::-1]):
        y_fig = (panel_top - header_h - n*row_slot) + ((i + 0.5) * row_slot)
        fig.text(LEFT, y_fig, lab, ha="left", va="center", color=LABEL_C, fontproperties=LABEL_FP)

    if show_xticks:
        trans = ax.get_xaxis_transform()
        offset_inner   = ScaledTranslation(7/72,0,fig.dpi_scale_trans)
        offset_pct_0   = ScaledTranslation(4/72,0,fig.dpi_scale_trans)
        offset_pct_100 = ScaledTranslation(10/72,0,fig.dpi_scale_trans)
        y_label = -0.075
        for gx in ticks:
            ax.plot([gx,gx],[-0.03,0.0], transform=trans, color=(0,0,0,0.6), lw=1.1, clip_on=False, zorder=4)
            ax.text(gx, y_label, f"{int(gx)}", transform=trans, ha="center", va="top", color="#000", fontproperties=TICK_FP, zorder=4, clip_on=False)
            if gx==0:   ax.text(gx, y_label, "%", transform=trans+offset_pct_0,   ha="left", va="top", color="#000", fontproperties=TICK_FP)
            elif gx==100: ax.text(gx, y_label, "%", transform=trans+offset_pct_100, ha="left", va="top", color="#000", fontproperties=TICK_FP)
            else:       ax.text(gx, y_label, "%", transform=trans+offset_inner,   ha="left", va="top", color="#000", fontproperties=TICK_FP)

    if draw_bottom_divider:
        y0 = (panel_top - header_h - n*row_slot) - 0.008
        fig.lines.append(plt.Line2D([LEFT, 1 - RIGHT], [y0, y0], transform=fig.transFigure, color=DIVIDER, lw=1.2, alpha=0.35))
    return (panel_top - header_h - n*row_slot) - GAP

# draw sections
y_top = 1 - TOP - header_block_h
for idx,(title,data) in enumerate(sections):
    is_last = idx == len(sections)-1
    y_top = draw_panel(y_top, title, data, show_xticks=is_last, draw_bottom_divider=not is_last)

# footer
fig.text((LEFT + gutter + (1 - RIGHT))/2.0, BOT * 0.1, footer_caption_text,
         ha="center", va="center", color="#222222", fontproperties=FOOTER_FP)

st.pyplot(fig, use_container_width=True)

# download
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
buf.seek(0)
st.download_button(
    "⬇️ Download Feature Z (PNG)",
    data=buf.getvalue(),
    file_name=f"{str(name_).replace(' ','_')}_featureZ.png",
    mime="image/png",
    key=f"download_feature_z_{uuid.uuid4().hex}"
)
import matplotlib.pyplot as _plt_cleanup
_plt_cleanup.close(fig)

# ======================== BUILD CONTEXT FOR AI REPORT ========================
# Create summary dictionaries used in the prompt

player_summary = {
    "Name": name_,
    "Age": age,
    "Team": team,
    "League": _safe_get(player_row, "League", "—"),
    "Position": pos,
    "Foot": foot_display,
    "Height": height_text if (show_height and height_text) else "—",
    "Minutes": minutes,
    "Goals": goals,
    "Assists": assists,
    "Role Fit Score": float(
        ranked.loc[ranked["Player"].astype(str) == str(name_), "Role Fit Score"]
        .head(1)
        .fillna(0)
        .values[0]
    )
    if "Role Fit Score" in ranked.columns
    else None,
    "Template League": template_league,
    "Template Team": template_team,
}

role_metrics = {
    m: (round(float(player_row[m].iloc[0]), 2) if m in player_row else None)
    for m in TEMPLATE_METRICS
    if m in player_row
}

def safe_pct(col):
    try:
        p = pct_series(col)
        return None if p is None or np.isnan(p) else round(float(p), 1)
    except Exception:
        return None

percentiles = {
    "Non-penalty goals per 90": safe_pct("Non-penalty goals per 90"),
    "xG per 90": safe_pct("xG per 90"),
    "Dribbles per 90": safe_pct("Dribbles per 90"),
    "Aerial duels per 90": safe_pct("Aerial duels per 90"),
    "Passes per 90": safe_pct("Passes per 90"),
    "Accurate passes, %": safe_pct("Accurate passes, %"),
}

# ======================== FREE AI SCOUTING REPORT via HUGGING FACE ========================
st.markdown("---")
st.header("🧠 AI Scouting Report (Free Model)")

include_online = st.checkbox("Include recent news context", value=True)
lookback_days = st.slider("Look-back (days)", 3, 60, 21, disabled=not include_online)

import requests, json, datetime as dt, re

HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
HEADERS = {"Authorization": f"Bearer {st.secrets.get('HF_API_KEY', '')}"}

def generate_report_hf(prompt: str) -> str:
    try:
        r = requests.post(HF_API_URL, headers=HEADERS, json={"inputs": prompt}, timeout=120)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
            return data[0]["generated_text"]
        return json.dumps(data, indent=2)
    except Exception as e:
        st.error(f"Hugging Face request failed: {e}")
        st.stop()

sections_req = """Include:
- Executive summary
- Technical profile (finishing, first touch, aerial, ball-carrying, passing/retention)
- Tactical fit
- Development & projection
- Suitability for higher tiers/leagues
500–1000 words.
"""

prompt = f"""
You are a professional football scout. Write a detailed, data-driven scouting report
based on the following player information and metrics.

PLAYER:
{json.dumps(player_summary, ensure_ascii=False)}

ROLE METRICS:
{json.dumps(role_metrics, ensure_ascii=False)}

KEY PERCENTILES (vs candidate pool):
{json.dumps(percentiles, ensure_ascii=False)}

{sections_req}
"""

with st.spinner("Generating scouting report (free model)…"):
    report = generate_report_hf(prompt)

st.markdown(report)
