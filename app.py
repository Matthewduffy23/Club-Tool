# app.py — Advanced Striker Scouting System (Role Template + Similar CFs + Feature Z)
# Requires: streamlit, pandas, numpy, matplotlib, pillow
# Run: streamlit run app.py

import io
import math
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
from matplotlib import font_manager as fm
from matplotlib.font_manager import FontProperties
from PIL import Image
from numpy.linalg import norm
from numpy import exp

# ----------------- PAGE -----------------
st.set_page_config(page_title="Advanced Striker Scouting System", layout="wide")
st.title("🔎 Advanced Striker Scouting System — CF Role Template")
st.caption("Type a team and league, pick a single player or use all CFs from that team as the template. We’ll find close role matches.")

# ----------------- DATA LOADER -----------------
@st.cache_data(show_spinner=False)
def _read_csv_from_path(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)

@st.cache_data(show_spinner=False)
def _read_csv_from_bytes(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data))

def load_df(csv_name: str = "WORLDJUNE25.csv") -> pd.DataFrame:
    """
    Looks for WORLDJUNE25.csv in common spots; falls back to uploader.
    """
    candidates = [
        Path.cwd() / csv_name,
        Path(__file__).resolve().parent.parent / csv_name,  # ../WORLDJUNE25.csv
        Path(__file__).resolve().parent / csv_name,         # ./WORLDJUNE25.csv
    ]
    for p in candidates:
        if p.exists():
            return _read_csv_from_path(str(p))

    st.warning(f"Could not find **{csv_name}**. Please upload below.")
    up = st.file_uploader("Upload WORLDJUNE25.csv", type=["csv"])
    if up is None:
        st.stop()
    return _read_csv_from_bytes(up.getvalue())

df = load_df("WORLDJUNE25.csv")

# ----------------- LEAGUES + RATINGS (your full lists) -----------------
INCLUDED_LEAGUES = [
    'England 1.', 'England 2.', 'England 3.', 'England 4.', 'England 5.',
    'England 6.', 'England 7.', 'England 8.', 'England 9.', 'England 10.',
    'Albania 1.', 'Algeria 1.', 'Andorra 1.', 'Argentina 1.', 'Armenia 1.',
    'Australia 1.', 'Austria 1.', 'Austria 2.', 'Azerbaijan 1.', 'Belgium 1.',
    'Belgium 2.', 'Bolivia 1.', 'Bosnia 1.', 'Brazil 1.', 'Brazil 2.', 'Brazil 3.',
    'Bulgaria 1.', 'Canada 1.', 'Chile 1.', 'Colombia 1.', 'Costa Rica 1.',
    'Croatia 1.', 'Cyprus 1.', 'Czech 1.', 'Czech 2.', 'Denmark 1.', 'Denmark 2.',
    'Ecuador 1.', 'Egypt 1.', 'Estonia 1.', 'Finland 1.', 'France 1.', 'France 2.',
    'France 3.', 'Georgia 1.', 'Germany 1.', 'Germany 2.', 'Germany 3.', 'Germany 4.',
    'Greece 1.', 'Hungary 1.', 'Iceland 1.', 'Israel 1.', 'Israel 2.', 'Italy 1.',
    'Italy 2.', 'Italy 3.', 'Japan 1.', 'Japan 2.', 'Kazakhstan 1.', 'Korea 1.',
    'Latvia 1.', 'Lithuania 1.', 'Malta 1.', 'Mexico 1.', 'Moldova 1.', 'Morocco 1.',
    'Netherlands 1.', 'Netherlands 2.', 'North Macedonia 1.', 'Northern Ireland 1.',
    'Norway 1.', 'Norway 2.', 'Paraguay 1.', 'Peru 1.', 'Poland 1.', 'Poland 2.',
    'Portugal 1.', 'Portugal 2.', 'Portugal 3.', 'Qatar 1.', 'Ireland 1.', 'Romania 1.',
    'Russia 1.', 'Saudi 1.', 'Scotland 1.', 'Scotland 2.', 'Scotland 3.', 'Serbia 1.',
    'Serbia 2.', 'Slovakia 1.', 'Slovakia 2.', 'Slovenia 1.', 'Slovenia 2.', 'South Africa 1.',
    'Spain 1.', 'Spain 2.', 'Spain 3.', 'Sweden 1.', 'Sweden 2.', 'Switzerland 1.',
    'Switzerland 2.', 'Tunisia 1.', 'Turkey 1.', 'Turkey 2.', 'Ukraine 1.', 'UAE 1.',
    'USA 1.', 'USA 2.', 'Uruguay 1.', 'Uzbekistan 1.', 'Venezuela 1.', 'Wales 1.'
]

PRESET_LEAGUES = {
    "Top 5 Europe": {'England 1.', 'France 1.', 'Germany 1.', 'Italy 1.', 'Spain 1.'},
    "Top 20 Europe": {
        'England 1.','Italy 1.','Spain 1.','Germany 1.','France 1.',
        'England 2.','Portugal 1.','Belgium 1.','Turkey 1.','Germany 2.','Spain 2.','France 2.',
        'Netherlands 1.','Austria 1.','Switzerland 1.','Denmark 1.','Croatia 1.','Italy 2.','Czech 1.','Norway 1.'
    },
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

# ----------------- CF FEATURES + BASE -----------------
FEATURES = [
    'Touches in box per 90', 'xG per 90',
    'Dribbles per 90', 'Progressive runs per 90',
    'Aerial duels per 90', 'Aerial duels won, %',
    'Passes per 90', 'Non-penalty goals per 90', 'Accurate passes, %'
]
REQUIRED_BASE = {"Player","Team","League","Age","Position","Minutes played","Market value","Goals"}

NEEDED = set(FEATURES) | REQUIRED_BASE
missing = [c for c in NEEDED if c not in df.columns]
if missing:
    st.error(f"Dataset missing required columns: {missing}")
    st.stop()

# Coerce numerics
for col in ["Minutes played","Age","Market value","Goals"] + FEATURES:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ----------------- SIDEBAR FILTERS -----------------
with st.sidebar:
    st.header("Filters")
    c1, c2, c3 = st.columns([1,1,1])
    use_top5  = c1.checkbox("Top-5 EU", value=False)
    use_top20 = c2.checkbox("Top-20 EU", value=False)
    use_efl   = c3.checkbox("EFL", value=False)

    seed = set()
    if use_top5:  seed |= PRESET_LEAGUES["Top 5 Europe"]
    if use_top20: seed |= PRESET_LEAGUES["Top 20 Europe"]
    if use_efl:   seed |= PRESET_LEAGUES["EFL (England 2–4)"]

    leagues_avail = sorted(set(INCLUDED_LEAGUES) | set(df.get("League", pd.Series([])).dropna().unique()))
    default_leagues = sorted(seed) if seed else INCLUDED_LEAGUES
    leagues_sel = st.multiselect("Leagues (add or prune the presets)", leagues_avail, default=default_leagues)

    # Position startswith
    pos_prefix = st.text_input("Position startswith", "CF")
    def position_filter(pos):
        return str(pos).strip().upper().startswith(str(pos_prefix).strip().upper())

    # Minutes / Age
    df["Minutes played"] = pd.to_numeric(df["Minutes played"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    min_minutes, max_minutes = st.slider("Minutes played", 0, 6000, (1000, 6000))
    age_min_data = int(np.nanmin(df["Age"])) if df["Age"].notna().any() else 14
    age_max_data = int(np.nanmax(df["Age"])) if df["Age"].notna().any() else 45
    min_age, max_age = st.slider("Age", age_min_data, age_max_data, (16, 50))

    # Contract (optional)
    apply_contract = st.checkbox("Filter by contract expiry", value=False)
    df["Contract expires"] = pd.to_datetime(df.get("Contract expires"), errors="coerce")
    cutoff_year = st.slider("Max contract year (inclusive)", 2025, 2030, 2026)

    # League strength
    min_strength, max_strength = st.slider("League quality (strength)", 0, 101, (0, 101))
    use_league_weighting = st.checkbox("Use league weighting in role score", value=False)
    beta = st.slider("League weighting beta", 0.0, 1.0, 0.40, 0.05, help="0 = ignore league strength; 1 = only league strength")

    # Market value
    df["Market value"] = pd.to_numeric(df["Market value"], errors="coerce")
    mv_col = "Market value"
    mv_max_raw = int(np.nanmax(df[mv_col])) if df[mv_col].notna().any() else 50_000_000
    mv_cap = int(math.ceil(mv_max_raw / 5_000_000) * 5_000_000)
    st.markdown("**Market value (€)**")
    use_m = st.checkbox("Adjust in millions", True)
    if use_m:
        max_m = int(mv_cap // 1_000_000)
        mv_min_m, mv_max_m = st.slider("Range (M€)", 0, max_m, (0, min(max_m, 10)))
        min_value = mv_min_m * 1_000_000
        max_value = mv_max_m * 1_000_000
    else:
        min_value, max_value = st.slider("Range (€)", 0, mv_cap, (0, min(mv_cap, 10_000_000)), step=100_000)

    # Role score strictness
    st.subheader("Role Score")
    decay_rate = st.slider("Exponential decay rate (higher = stricter)", 0.5, 10.0, 5.0, 0.5)

    # Table size
    top_n = st.number_input("Top N (table)", 5, 200, 50, 5)

# ----------------- FILTER POOL -----------------
df_f = df[df["League"].isin(leagues_sel)].copy()
df_f = df_f[df_f["Position"].astype(str).apply(position_filter)]
df_f = df_f[df_f["Minutes played"].between(min_minutes, max_minutes)]
df_f = df_f[df_f["Age"].between(min_age, max_age)]
if apply_contract and "Contract expires" in df_f.columns:
    df_f = df_f[df_f["Contract expires"].dt.year <= cutoff_year]
df_f["League Strength"] = df_f["League"].map(LEAGUE_STRENGTHS).fillna(0.0)
df_f = df_f[(df_f["League Strength"] >= float(min_strength)) & (df_f["League Strength"] <= float(max_strength))]
df_f = df_f[(df_f["Market value"] >= min_value) & (df_f["Market value"] <= max_value)]
df_f = df_f.dropna(subset=FEATURES)

if df_f.empty:
    st.warning("No players after filters. Loosen filters.")
    st.stop()

# ----------------- TEMPLATE PICKER -----------------
st.markdown("---")
st.header("🎯 Choose Role Template (Team & Optional Single Player)")

all_teams = sorted(df_f["Team"].dropna().astype(str).unique())
team_default = all_teams[0] if all_teams else ""
template_team = st.text_input("Team (type exact)", value=team_default, placeholder="e.g., Southampton")

leagues_for_team = sorted(df_f.loc[df_f["Team"].astype(str)==template_team, "League"].dropna().astype(str).unique())
league_choices = leagues_for_team if leagues_for_team else sorted(df_f["League"].dropna().astype(str).unique())
template_league = st.selectbox("League for template team", league_choices, index=0 if league_choices else None)

use_single_template_player = st.checkbox("Use single player only (otherwise averages all CFs at team)", value=True)

players_in_team = []
if template_team and template_league:
    mask_tl = (
        (df_f["Team"].astype(str) == template_team) &
        (df_f["League"].astype(str) == template_league)
    )
    players_in_team = sorted(df_f.loc[mask_tl, "Player"].dropna().astype(str).unique())

template_player_name = st.selectbox(
    "Template player (if single)",
    options=["— Select a player —"] + players_in_team if players_in_team else ["— none available —"],
    index=0
)

# ----------------- ROLE METRICS + TEMPLATE -----------------
cf_df = df_f.copy()
cf_df["Opportunities"]      = 0.7*cf_df['Touches in box per 90'] + 0.3*cf_df['xG per 90']
cf_df["Ball Carrying"]      = 0.65*cf_df['Dribbles per 90'] + 0.35*cf_df['Progressive runs per 90']
cf_df["Aerial Requirement"] = cf_df['Aerial duels per 90'] * cf_df['Aerial duels won, %'] / 100.0
cf_df["Passing Volume"]     = cf_df['Passes per 90']
cf_df["Goal Output"]        = cf_df['Non-penalty goals per 90']
cf_df["Retention"]          = cf_df['Accurate passes, %']

TEMPLATE_METRICS = ["Opportunities","Ball Carrying","Aerial Requirement","Passing Volume","Goal Output","Retention"]

if not template_team or not template_league:
    st.info("Pick a template team & league to continue.")
    st.stop()

if use_single_template_player:
    template_df = cf_df[
        (cf_df["Team"].astype(str) == template_team) &
        (cf_df["League"].astype(str) == template_league) &
        (cf_df["Player"].astype(str) == (template_player_name if (template_player_name and not template_player_name.startswith("—")) else ""))
    ].copy()
else:
    template_df = cf_df[
        (cf_df["Team"].astype(str) == template_team) &
        (cf_df["League"].astype(str) == template_league)
    ].copy()

if use_single_template_player and template_df.empty:
    st.error("No CFs for that (team, league, player). Check spelling/filters or untick single-player.")
    st.stop()
if (not use_single_template_player) and template_df.empty:
    st.error("No CFs for that team & league under current pool filters.")
    st.stop()

st.subheader("🧩 Players used for Role Template")
st.dataframe(
    template_df[["Player","Minutes played","Position","League"]].sort_values("Minutes played", ascending=False),
    use_container_width=True
)

template_vector = template_df[TEMPLATE_METRICS].mean()

# Remove template team+league players from comparison pool
comparison_df = cf_df[
    ~((cf_df["Team"].astype(str) == template_team) & (cf_df["League"].astype(str) == template_league))
].copy()

# Scouting constraints (match your snippet)
comparison_df = comparison_df[
    (comparison_df["Age"] <= 26) &
    (comparison_df["Market value"] <= 10_000_000) &
    (comparison_df["Minutes played"] >= 1000)
].copy()

if comparison_df.empty:
    st.warning("No candidates after constraints (Age ≤26, MV ≤10M, Min≥1000). Loosen constraints.")
    st.stop()

# Role fit distance + score
def _row_dist(row):
    return norm([
        row['Opportunities']      - template_vector['Opportunities'],
        row['Ball Carrying']      - template_vector['Ball Carrying'],
        row['Aerial Requirement'] - template_vector['Aerial Requirement'],
        row['Passing Volume']     - template_vector['Passing Volume'],
        row['Goal Output']        - template_vector['Goal Output'],
        row['Retention']          - template_vector['Retention'],
    ])

comparison_df["Role Fit Distance"] = comparison_df.apply(_row_dist, axis=1)

# Optional league weighting (beta blend with normalized league strength)
if use_league_weighting:
    ls = comparison_df["League"].map(LEAGUE_STRENGTHS).fillna(0.0) / 100.0
else:
    ls = 0.0

min_dist = float(comparison_df["Role Fit Distance"].min())
max_dist = float(comparison_df["Role Fit Distance"].max())
rng = max_dist - min_dist
if rng <= 1e-12:
    base_score = pd.Series(100.0, index=comparison_df.index)
else:
    base_score = 100.0 * exp(-decay_rate * ((comparison_df["Role Fit Distance"] - min_dist) / rng))

# Blend: (1-beta)*role_score + beta*(league_strength*100)
if isinstance(ls, pd.Series):
    league_part = (ls * 100.0)
else:
    league_part = 0.0
comparison_df["Role Fit Score"] = (1.0 - beta) * base_score + beta * league_part

ranked = comparison_df.sort_values(by="Role Fit Score", ascending=False).reset_index(drop=True)

st.markdown("---")
st.header("🏅 Top Role Matches")
cols_to_show = ["Player","Team","League","Age","Minutes played","Market value","Role Fit Score"]
st.dataframe(ranked[cols_to_show].head(int(top_n)), use_container_width=True)

# ----------------- FEATURE Z (underneath table) -----------------
st.markdown("---")
st.header("📋 Feature Z — White Percentile Board")

# ======== Feature Z helpers (percentiles & value formatting) ========
POLAR_METRICS = [
    "Non-penalty goals per 90","xG per 90","Shots per 90",
    "Dribbles per 90","Passes to penalty area per 90","Touches in box per 90",
    "Aerial duels per 90","Aerial duels won, %","Passes per 90",
    "Accurate passes, %","xA per 90","Progressive runs per 90",
]

# ensure presence if some of these aren't in dataset; filter to existing
POLAR_METRICS = [m for m in POLAR_METRICS if m in df_f.columns]

# Create a quick picker — default to best match
left, right = st.columns([2,2])
with left:
    options_ranked = ranked["Player"].astype(str).head(int(top_n)).tolist()
    any_pool = st.checkbox("Pick from entire filtered pool (not just Top N)", value=False)
    if any_pool:
        options = sorted(df_f["Player"].dropna().astype(str).unique())
    else:
        options = options_ranked
    player_sel = st.selectbox("Choose player for Feature Z", options, index=0 if options else None)

with right:
    show_height = st.checkbox("Show height in info row", value=True)
    foot_override_on = st.checkbox("Edit foot value", value=False)
    foot_override_text = st.text_input("Foot (e.g., Left)", value="", disabled=not foot_override_on)
    name_override_on = st.checkbox("Edit display name", value=False)
    name_override = st.text_input("Display name", "", disabled=not name_override_on)

# Build player_row
player_row = df_f[df_f["Player"].astype(str) == str(player_sel)].head(1)
if player_row.empty:
    st.info("Pick a player above.")
    st.stop()

# Percentile helper
def pct_series(col: str) -> float:
    vals = pd.to_numeric(df_f[col], errors="coerce").dropna()
    if vals.empty: return np.nan
    v = pd.to_numeric(player_row.iloc[0][col], errors="coerce")
    if pd.isna(v): return np.nan
    # percentile rank (inclusive)
    pct = (vals <= v).mean() * 100.0
    return float(pct)

def val_of(col: str):
    v = player_row.iloc[0].get(col)
    if pd.isna(v): return (np.nan, "—")
    if isinstance(v, (int,float,np.floating)):
        # simple formatting
        if "%" in col:
            return (float(v), f"{float(v):.0f}%")
        else:
            return (float(v), f"{float(v):.2f}")
    return (v, str(v))

# ======== Feature Z controls (images & caption) ========
with st.expander("Feature Z options", expanded=False):
    enable_images = st.checkbox("Add header images", value=True)
    _CAPTION_DEFAULT = "Percentile Rank"
    _edit_footer = st.toggle("Edit footer caption", value=False, key="fz_edit_footer")
    footer_caption_text = st.text_input("Footer caption", _CAPTION_DEFAULT, disabled=not _edit_footer, key="fz_footer_text")

    st.caption("Upload up to three header images (PNG recommended). Rightmost is the anchor.")
    up_img1 = st.file_uploader("Image 1 (rightmost)", type=["png","jpg","jpeg","webp"], key="fz_img1") if enable_images else None
    up_img2 = st.file_uploader("Image 2 (middle)",   type=["png","jpg","jpeg","webp"], key="fz_img2") if enable_images else None
    up_img3 = st.file_uploader("Image 3 (leftmost)", type=["png","jpg","jpeg","webp"], key="fz_img3") if enable_images else None

# ======== Feature Z data assembly (sections) ========
def _safe_get(sr, key, default="—"):
    try:
        v = sr.iloc[0].get(key, default)
        s = "" if v is None else str(v)
        return default if s.strip() == "" else s
    except Exception:
        return default

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
    if col in player_row.columns and str(_safe_get(player_row, col, "")).strip():
        height_text = str(_safe_get(player_row, col, "")).strip()
        break

# Sections (Attacking/Defensive/Possession) using POLAR_METRICS where applicable
ATTACKING = []
for lab, met in [
    ("Goals: Non-Penalty","Non-penalty goals per 90"),
    ("xG","xG per 90"),
    ("Shots","Shots per 90"),
    ("Header Goals","Head goals per 90"),
    ("Expected Assists","xA per 90"),
    ("Progressive Runs","Progressive runs per 90"),
    ("Touches in Opp. Box","Touches in box per 90"),
]:
    if met in df_f.columns:
        ATTACKING.append((lab, float(np.nan_to_num(pct_series(met), nan=0.0)), val_of(met)[1]))

DEFENSIVE = []
for lab, met in [
    ("Aerial Duels","Aerial duels per 90"),
    ("Aerial Duel Success %","Aerial duels won, %"),
    ("PAdj. Interceptions","PAdj Interceptions"),
    ("Defensive Duels","Defensive duels per 90"),
    ("Defensive Duel Success %","Defensive duels won, %"),
]:
    if met in df_f.columns:
        DEFENSIVE.append((lab, float(np.nan_to_num(pct_series(met), nan=0.0)), val_of(met)[1]))

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
    if met in df_f.columns:
        POSSESSION.append((lab, float(np.nan_to_num(pct_series(met), nan=0.0)), val_of(met)[1]))

sections = [("Attacking",ATTACKING),("Defensive",DEFENSIVE),("Possession",POSSESSION)]
sections = [(t,lst) for t,lst in sections if lst]

# ======== Feature Z style & drawing ========
# Fonts (fallbacks)
def _font_name_or_fallback(pref, fallback="DejaVu Sans"):
    installed = {f.name for f in fm.fontManager.ttflist}
    for n in pref:
        if n in installed: return n
    return fallback

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
    # red → gold → green blend
    v=float(np.clip(v,0,100))
    TAB_RED=np.array([199,54,60]); TAB_GOLD=np.array([240,197,106]); TAB_GREEN=np.array([61,166,91])
    def _blend(c1,c2,t): c=c1+(c2-c1)*np.clip(t,0,1); return f"#{int(c[0]):02x}{int(c[1]):02x}{int(c[2]):02x}"
    return _blend(TAB_RED,TAB_GOLD,v/50) if v<=50 else _blend(TAB_GOLD,TAB_GREEN,(v-50)/50)

def _open_upload(u):
    if u is None: return None
    try: return Image.open(u).convert("RGBA")
    except Exception: return None

fig_size   = (11.8, 9.6) if True else (10, 8)
dpi = 120
title_row_h = 0.125
header_block_h = title_row_h + 0.055 if True else (0.075 + 0.020)
img_box_w = img_box_h = 0.16
img_gap = 0.003

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

row1 = [("Position: ",pos), ("Age: ",age), ("Height: ", (height_text if (show_height and height_text) else "—"))]
row2 = [("Games: ",games), ("Goals: ",goals), ("Assists: ",assists)]
row3 = [("Minutes: ",minutes), ("Foot: ",foot_display)]

title_y = 1 - TOP - 0.010
y1 = title_y - 0.055
y2 = y1 - 0.039
y3 = y2 - 0.039
draw_pairs_line(row1, y1); draw_pairs_line(row2, y2); draw_pairs_line(row3, y3)

# Images (optional)
if up_img1 or up_img2 or up_img3:
    def add_header_image(pil_img, right_index=0):
        if pil_img is None: return
        x_right_edge = 1 - RIGHT
        x = x_right_edge - (right_index + 1) * img_box_w - right_index * img_gap
        y_top_band = 1 - TOP - 0.006
        y = y_top_band - img_box_h
        ax_img = fig.add_axes([x, y, img_box_w, img_box_h])
        ax_img.imshow(pil_img); ax_img.axis("off")
    add_header_image(_open_upload(up_img1), right_index=0)
    add_header_image(_open_upload(up_img2), right_index=1)
    add_header_image(_open_upload(up_img3), right_index=2)

# Divider
fig.lines.append(plt.Line2D([LEFT, 1 - RIGHT],
                            [1 - TOP - header_block_h + 0.004]*2,
                            transform=fig.transFigure, color=DIVIDER, lw=0.8, alpha=0.35))

# Panels
def draw_panel(panel_top, title, tuples, *, show_xticks=False, draw_bottom_divider=True):
    n = len(tuples)
    if n == 0: return panel_top
    # compute space
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

    # tracks + grid
    for i in range(n):
        ax.add_patch(plt.Rectangle((0, i-(BAR_FRAC/2)), 100, BAR_FRAC, color=TRACK, ec="none", zorder=0.5))
    for gx in ticks:
        ax.vlines(gx, -0.5, n-0.5, colors=(0,0,0,0.16), linewidth=0.8, zorder=0.75)

    # bars (reverse to list top-to-bottom)
    for i,(lab,pct,val_str) in enumerate(tuples[::-1]):
        y = i; bar_w = float(np.clip(pct,0,100))
        ax.add_patch(plt.Rectangle((0, y-(BAR_FRAC/2)), bar_w, BAR_FRAC, color=pct_to_rgb(bar_w), ec="none", zorder=1.0))
        x_text = 1.0 if bar_w >= 3 else min(100.0, bar_w + 0.8)
        ax.text(x_text, y, val_str, ha="left", va="center", color="#0B0B0B", fontproperties=BAR_VALUE_FP, zorder=2.0, clip_on=False)

    # 50% line
    ax.axvline(50, color="#000000", ls=(0,(4,4)), lw=1.5, alpha=0.7, zorder=3.5)

    # left labels
    for i,(lab,_,_) in enumerate(tuples[::-1]):
        y_fig = (panel_top - header_h - n*row_slot) + ((i + 0.5) * row_slot)
        fig.text(LEFT, y_fig, lab, ha="left", va="center", color=LABEL_C, fontproperties=LABEL_FP)

    # x ticks bottom
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

    # divider
    if draw_bottom_divider:
        y0 = (panel_top - header_h - n*row_slot) - 0.008
        fig.lines.append(plt.Line2D([LEFT, 1 - RIGHT], [y0, y0], transform=fig.transFigure, color=DIVIDER, lw=1.2, alpha=0.35))
    return (panel_top - header_h - n*row_slot) - GAP

# Build tuples with percentiles from df_f
def make_tuples(pairs):
    out = []
    for lab, met in pairs:
        if met in df_f.columns:
            out.append((lab, float(np.nan_to_num(pct_series(met), nan=0.0)), val_of(met)[1]))
    return out

# Use already-built ATTACKING/DEFENSIVE/POSSESSION
sections = [("Attacking", ATTACKING), ("Defensive", DEFENSIVE), ("Possession", POSSESSION)]
sections = [(t,lst) for t,lst in sections if lst]

# Layout draw
y_top = 1 - TOP - header_block_h
for idx,(title,data) in enumerate(sections):
    is_last = idx == len(sections)-1
    y_top = draw_panel(y_top, title, data, show_xticks=is_last, draw_bottom_divider=not is_last)

# Footer
fig.text((LEFT + gutter + (1 - RIGHT))/2.0, BOT * 0.1, st.session_state.get("fz_footer_text", "Percentile Rank"),
         ha="center", va="center", color=LABEL_C, fontproperties=FOOTER_FP)

st.pyplot(fig, use_container_width=True)

# Download
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
plt.close(fig)

# ----------------- END -----------------
