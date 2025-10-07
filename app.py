# app.py — Role Template CF Scouting + Feature Z (fixed template scope, league→team flow, league weighting)
# Run: streamlit run app.py

import io, math, uuid
from pathlib import Path
import numpy as np, pandas as pd, streamlit as st
from numpy.linalg import norm
from numpy import exp

st.set_page_config(page_title="Advanced Striker Scouting System", layout="wide")
st.title("🔎 Advanced Striker Scouting System — CF Role Template")
st.caption("Pick a league → team (and optional player) for the template. Template is sourced from RAW data; pool filters only shape the matches.")

# ----------------- CSV loader (safe) -----------------
@st.cache_data(show_spinner=False)
def _read_csv_from_path(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)

@st.cache_data(show_spinner=False)
def _read_csv_from_bytes(data: bytes) -> pd.DataFrame:
    import io
    return pd.read_csv(io.BytesIO(data))

def load_df(csv_name: str = "WORLDJUNE25.csv") -> pd.DataFrame:
    candidates = [Path.cwd() / csv_name, Path.cwd().parent / csv_name]
    try:
        here = Path(__file__).resolve().parent
        candidates += [here / csv_name, here.parent / csv_name]
    except Exception:
        pass
    for p in candidates:
        if p.exists(): return _read_csv_from_path(str(p))
    st.warning(f"Could not find **{csv_name}**. Please upload below.")
    up = st.file_uploader("Upload WORLDJUNE25.csv", type=["csv"])
    if up is None: st.stop()
    return _read_csv_from_bytes(up.getvalue())

df = load_df("WORLDJUNE25.csv")

# ----------------- leagues & strengths -----------------
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

# ----------------- columns & coercions -----------------
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

# ----------------- SIDEBAR: candidate pool filters -----------------
with st.sidebar:
    st.header("Candidate Pool Filters (affect MATCHES, not the TEMPLATE)")
    # Presets
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

    # Role score settings
    st.subheader("Role Score")
    decay_rate = st.slider("Exp. decay (↑=stricter)", 0.5, 10.0, 5.0, 0.5)
    use_league_weighting = st.checkbox("Blend in league strength (β)", value=False)
    beta = st.slider("β (0–1)", 0.0, 1.0, 0.40, 0.05, help="0=distance only, 1=league strength only")

    top_n = st.number_input("Top N (table)", 5, 200, 50, 5)

# --------- Candidate pool (affected by sidebar filters) ---------
df_pool = df[df["League"].isin(leagues_sel)].copy()
df_pool = df_pool[df_pool["Position"].apply(position_filter)]
df_pool = df_pool[df_pool["Minutes played"].between(min_minutes, max_minutes)]
df_pool = df_pool[df_pool["Age"].between(min_age, max_age)]
df_pool = df_pool[df_pool["Market value"].between(pool_min_value, pool_max_value)]
df_pool = df_pool.dropna(subset=FEATURES)
df_pool["League Strength"] = df_pool["League"].map(LEAGUE_STRENGTHS).fillna(0.0)

if df_pool.empty:
    st.warning("No players in candidate pool after filters. Loosen filters.")
    st.stop()

# ----------------- TEMPLATE SECTION (NOT affected by pool filters) -----------------
st.markdown("---")
st.header("🎯 Template selection (RAW data) — League → Team → Player (optional)")

# 1) choose template league from ALL leagues present in RAW df
template_league_list = sorted([str(x) for x in df["League"].dropna().unique()])
template_league = st.selectbox("Template league (scopes team list)", template_league_list)

# 2) team search + select (only teams IN that league, from RAW df)
templ_teams_all = sorted(df.loc[df["League"].astype(str) == template_league, "Team"].dropna().astype(str).unique())
search = st.text_input("Search team (filters list)", "")
templ_teams = [t for t in templ_teams_all if search.lower() in t.lower()] or templ_teams_all
template_team = st.selectbox("Template team", templ_teams)

# 3) optional single player (from RAW df, CF only + min minutes threshold you choose)
min_minutes_template = st.slider("Minimum minutes for template CFs", 0, 6000, 1000, 100)
use_single_template_player = st.checkbox("Use single player only (else avg of team CFs)", True)

# Template source (RAW df, not filtered), only CF & minutes threshold
tmpl_pos_ok = lambda p: str(p).strip().upper().startswith("CF")
df_template_source = df[
    (df["League"].astype(str) == template_league) &
    (df["Team"].astype(str) == template_team) &
    (df["Position"].apply(tmpl_pos_ok)) &
    (pd.to_numeric(df["Minutes played"], errors="coerce") >= min_minutes_template)
].copy().dropna(subset=FEATURES)

# single player dropdown
players_in_team_raw = sorted(df_template_source["Player"].dropna().astype(str).unique())
template_player_name = st.selectbox(
    "Template player (if single)",
    options=["— Select a player —"] + players_in_team_raw if players_in_team_raw else ["— none available —"],
    index=0
)

# ----------------- ROLE METRICS -----------------
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

st.subheader("🧩 Players used for Role Template (RAW, unaffected by pool filters)")
st.dataframe(
    template_df[["Player","Minutes played","Position","League"]].sort_values("Minutes played", ascending=False),
    use_container_width=True
)

template_vector = template_df[TEMPLATE_METRICS].mean()

# ----------------- MATCHING (candidate pool) -----------------
cf_pool = add_role_metrics(df_pool)
# exclude template team+league from pool matches (so you don't 'find yourself')
cf_pool = cf_pool[~((cf_pool["Team"].astype(str) == template_team) & (cf_pool["League"].astype(str) == template_league))].copy()

# hard scouting caps (same as before)
cf_pool = cf_pool[(cf_pool["Age"] <= 26) & (cf_pool["Market value"] <= 10_000_000) & (cf_pool["Minutes played"] >= 1000)]
if cf_pool.empty:
    st.warning("No candidates after age/value/minutes caps. Loosen pool filters or caps.")
    st.stop()

def row_dist(row):
    return norm([
        row['Opportunities']      - template_vector['Opportunities'],
        row['Ball Carrying']      - template_vector['Ball Carrying'],
        row['Aerial Requirement'] - template_vector['Aerial Requirement'],
        row['Passing Volume']     - template_vector['Passing Volume'],
        row['Goal Output']        - template_vector['Goal Output'],
        row['Retention']          - template_vector['Retention'],
    ])

cf_pool["Role Fit Distance"] = cf_pool.apply(row_dist, axis=1)

# base exp-decay score
min_d = float(cf_pool["Role Fit Distance"].min()); max_d = float(cf_pool["Role Fit Distance"].max())
rng = max_d - min_d
if rng <= 1e-12:
    base_score = pd.Series(100.0, index=cf_pool.index)
else:
    base_score = 100.0 * exp(-decay_rate * ((cf_pool["Role Fit Distance"] - min_d) / rng))

# -------- League weighting (candidate ONLY) --------
# RoleFit = (1−β)*base_score + β*(LeagueStrength%)
if use_league_weighting:
    league_part = cf_pool["League"].map(LEAGUE_STRENGTHS).fillna(0.0)
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

# ----------------- FEATURE Z under the table -----------------
st.markdown("---")
st.header("📋 Feature Z — White Percentile Board")
# (lazy import heavy libs so the app paints fast)
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
from matplotlib import font_manager as fm
from matplotlib.font_manager import FontProperties
from PIL import Image

# player picker (defaults to best match)
left, right = st.columns([2,2])
with left:
    options_ranked = ranked["Player"].astype(str).head(int(top_n)).tolist()
    any_pool = st.checkbox("Pick from entire candidate pool (not just Top N)", value=False)
    options = sorted(df_pool["Player"].dropna().astype(str).unique()) if any_pool else options_ranked
    if not options: st.stop()
    player_sel = st.selectbox("Choose player for Feature Z", options, index=0)
with right:
    show_height = st.checkbox("Show height in info row", value=True)
    foot_override_on = st.checkbox("Edit foot value", value=False)
    foot_override_text = st.text_input("Foot (e.g., Left)", value="", disabled=not foot_override_on)
    name_override_on = st.checkbox("Edit display name", value=False)
    name_override = st.text_input("Display name", "", disabled=not name_override_on)
    footer_caption_text = st.text_input("Footer caption", "Percentile Rank")

player_row = df_pool[df_pool["Player"].astype(str) == str(player_sel)].head(1)
if player_row.empty:
    st.info("Pick a player above.")
    st.stop()

# --- (Feature Z drawing code unchanged from previous safe version) ---
# ... to keep this message short, reuse the exact Feature Z block you already have below this comment ...
# It reads from df_pool for percentiles, uses name_/team/age/goals/etc., and ends with the PNG download.
# If you want me to paste the full block again, say the word and I'll drop it in verbatim.

# (mini helper so you can paste your existing Feature Z right here)
def _safe_get(sr, key, default="—"):
    try:
        v = sr.iloc[0].get(key, default)
        s = "" if v is None else str(v)
        return default if s.strip() == "" else s
    except Exception:
        return default

# ---- paste your existing Feature Z code here (from the last version I sent) ----
# (It will work as-is because we preserved variable names used by that block.)



