# Club Fit Tiles App — 5-role tabs (ST / Wingers / CM / FB / CB)
# - Keeps your glossy tiles + photo override + Feature Z
# - Single "Club Selection" for team/league, but each tab computes its own
#   role features, template vector, candidate pool, and Fit% independently.

import io, math, uuid, re, time
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import streamlit as st
from numpy.linalg import norm
from numpy import exp

# ---------- Page ----------
st.set_page_config(page_title="Club Scouting — Tiles", layout="wide")
st.title("🔎 Advanced Club Scouting — Tiles View")
st.caption(
    "Club Selection → Role Template matching with glossy tiles per role. "
    "Each tab computes its own Fit %. Dropdown per tile lets you paste a custom image URL to override the photo."
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

# ======================== sidebar: candidate pool filters (generic) ========================
with st.sidebar:
    st.header("Candidate Pool Filters (affect MATCHES)")
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

    # NOTE: per-role position filters are inside each tab
    min_minutes, max_minutes = st.slider("Minutes played (pool)", 0, 6000, (750, 6000))
    age_min_data = int(np.nanmin(pd.to_numeric(df["Age"], errors="coerce"))) if df["Age"].notna().any() else 14
    age_max_data = int(np.nanmax(pd.to_numeric(df["Age"], errors="coerce"))) if df["Age"].notna().any() else 50
    min_age, max_age = st.slider("Age (pool)", age_min_data, age_max_data, (16, 50))

    st.markdown("**Market value (€)**")
    mv_col = pd.to_numeric(df["Market value"], errors="coerce")
    mv_max_raw = int(np.nanmax(mv_col)) if mv_col.notna().any() else 50_000_000
    mv_cap = int(math.ceil(mv_max_raw / 5_000_000) * 5_000_000)
    use_m = st.checkbox("Adjust in millions", True)
    if use_m:
        max_m = max(1, mv_cap // 1_000_000)
        mv_min_m, mv_max_m = st.slider("Range (M€)", 0, max_m, (0, min(max_m, 10)))
        pool_min_value, pool_max_value = mv_min_m*1_000_000, mv_max_m*1_000_000
    else:
        pool_min_value, pool_max_value = st.slider("Range (€)", 0, mv_cap, (0, min(mv_cap, 10_000_000)), step=100_000)

    min_strength, max_strength = st.slider(
        "League quality (strength)", 0, 101, (0, 101),
        help="Filter candidates by league strength (0–100)."
    )

    # Role score settings
    st.subheader("Role Score")
    decay_rate = st.slider("Exp. decay (↑=stricter)", 0.5, 10.0, 5.0, 0.5)

    use_league_weighting = st.checkbox("Blend in league strength (β)", value=True)
    beta = st.slider("β (0–1)", 0.0, 1.0, 0.40, 0.05, help="0=distance only, 1=league strength only")

    use_league_mismatch = st.checkbox(
        "Penalise league mismatch inside distance (α, p)", value=True,
        help="Adds a distance-penalty based on |candidate league − template league|."
    )
    alpha = st.slider("League mismatch weight α", 0.0, 5.0, 1.20, 0.05)
    p_exp = st.slider("League mismatch exponent p", 1.0, 3.0, 1.50, 0.10)
    penalty_mode = st.selectbox("Penalty combine mode", ["Additive (stronger)", "Quadrature (gentler)"], index=0)

    top_n = st.number_input("How many tiles (Top N)", 5, 200, 20, 5)
    DEBUG_PHOTOS = st.checkbox("Debug player photos", False)

# ======================== helpers: build base candidate pool (no position) ========================
def build_base_pool():
    p = df.copy()
    p = p[p["League"].isin(leagues_sel)]
    # numeric coercions
    for c in ["Minutes played","Age","Market value","Goals"]:
        p[c] = pd.to_numeric(p[c], errors="coerce")
    p = p[p["Minutes played"].between(min_minutes, max_minutes)]
    p = p[p["Age"].between(min_age, max_age)]
    p = p[p["Market value"].between(pool_min_value, pool_max_value)]
    p["League Strength"] = p["League"].map(LEAGUE_STRENGTHS).fillna(0.0)
    p = p[(p["League Strength"] >= float(min_strength)) & (p["League Strength"] <= float(max_strength))]
    return p

# ======================== Club Selection (re-used by all tabs) ========================
st.markdown("---")
st.header("🎯 Club Selection (template source)")

template_league_list = sorted([str(x) for x in df["League"].dropna().unique()])
template_league = st.selectbox("Template league (scopes team list)", template_league_list)

templ_teams_all = sorted(df.loc[df["League"].astype(str) == template_league, "Team"].dropna().astype(str).unique())
search = st.text_input("Search team (filters list)", "")
templ_teams = [t for t in templ_teams_all if search.lower() in t.lower()] or templ_teams_all
template_team = st.selectbox("Template team", templ_teams)

min_minutes_template = st.slider("Minimum minutes for template players", 0, 6000, 1000, 100)
use_single_template_player = st.checkbox("Use single player only (else avg of role at team)", False)

template_strength = float(LEAGUE_STRENGTHS.get(template_league, 0.0))

# ======================== scoring helper used by all roles ========================
def _score_block(df_with_baseDist: pd.DataFrame) -> pd.DataFrame:
    """Given df with ['BaseDist','League'] columns, compute Role Fit Score with options."""
    # league mismatch inside distance
    if use_league_mismatch:
        base_min, base_max = float(df_with_baseDist["BaseDist"].min()), float(df_with_baseDist["BaseDist"].max())
        spread = max(1e-9, base_max - base_min)

        def _with_pen(row):
            ls = float(LEAGUE_STRENGTHS.get(str(row["League"]), 0.0))
            delta = abs(ls - template_strength) / 100.0
            pen = alpha * (delta ** p_exp) * spread
            return row["BaseDist"] + pen if penalty_mode.startswith("Additive") else float(np.hypot(row["BaseDist"], pen))
        df_with_baseDist["Role Fit Distance"] = df_with_baseDist.apply(_with_pen, axis=1)
    else:
        df_with_baseDist["Role Fit Distance"] = df_with_baseDist["BaseDist"]

    # exp-decay base score
    dmin, dmax = float(df_with_baseDist["Role Fit Distance"].min()), float(df_with_baseDist["Role Fit Distance"].max())
    rng = dmax - dmin
    if rng <= 1e-12:
        base_score = pd.Series(100.0, index=df_with_baseDist.index)
    else:
        base_score = 100.0 * exp(-decay_rate * ((df_with_baseDist["Role Fit Distance"] - dmin) / rng))

    league_part = df_with_baseDist["League"].map(LEAGUE_STRENGTHS).fillna(0.0) if use_league_weighting else 0.0
    df_with_baseDist["Role Fit Score"] = (1.0 - beta) * base_score + beta * league_part
    return df_with_baseDist.sort_values("Role Fit Score", ascending=False).reset_index(drop=True)

# ======================== role calculators ========================
def _template_rows_for_role(pos_predicate):
    src = df[
        (df["League"].astype(str) == template_league) &
        (df["Team"].astype(str) == template_team) &
        (df["Position"].apply(lambda p: pos_predicate(str(p))))
    ].copy()
    src["Minutes played"] = pd.to_numeric(src["Minutes played"], errors="coerce")
    src = src[src["Minutes played"] >= min_minutes_template]
    return src

def compute_strikers():
    # features + metrics (your original ST)
    feats = ['Touches in box per 90','xG per 90','Dribbles per 90','Progressive runs per 90',
             'Aerial duels per 90','Aerial duels won, %','Passes per 90','Non-penalty goals per 90','Accurate passes, %']

    tmpl_src = _template_rows_for_role(lambda p: p.strip().upper().startswith("CF")).dropna(subset=feats)
    if use_single_template_player:
        players = sorted(tmpl_src["Player"].dropna().astype(str).unique())
        chosen = st.selectbox("Template player (ST)", ["— Select —"] + players, index=0, key="st_tmpl_pick")
        if chosen and not chosen.startswith("—"):
            tmpl_src = tmpl_src[tmpl_src["Player"].astype(str) == chosen]
    if tmpl_src.empty:
        st.error("No strikers found for template conditions."); st.stop()

    f = tmpl_src.copy()
    f["Opportunities"]      = 0.7*f['Touches in box per 90'] + 0.3*f['xG per 90']
    f["Ball Carrying"]      = 0.65*f['Dribbles per 90'] + 0.35*f['Progressive runs per 90']
    f["Aerial Requirement"] = f['Aerial duels per 90'] * f['Aerial duels won, %'] / 100.0
    f["Passing Volume"]     = f['Passes per 90']
    f["Goal Output"]        = f['Non-penalty goals per 90']
    f["Retention"]          = f['Accurate passes, %']
    tmpl_vec = f[["Opportunities","Ball Carrying","Aerial Requirement","Passing Volume","Goal Output","Retention"]].mean()

    base_pool = build_base_pool()
    pool = base_pool.copy()
    pool = pool[pool["Position"].apply(lambda p: str(p).strip().upper().startswith("CF"))]
    pool = pool[~((pool["Team"].astype(str) == template_team) & (pool["League"].astype(str) == template_league))].copy()
    # hard caps for ST
    pool = pool[(pd.to_numeric(pool["Age"], errors="coerce") <= 26) &
                (pd.to_numeric(pool["Market value"], errors="coerce") <= 10_000_000) &
                (pd.to_numeric(pool["Minutes played"], errors="coerce") >= 1000)]
    # compute metrics on pool
    for c in feats:
        pool[c] = pd.to_numeric(pool[c], errors="coerce")
    pool = pool.dropna(subset=feats)
    pool["Opportunities"]      = 0.7*pool['Touches in box per 90'] + 0.3*pool['xG per 90']
    pool["Ball Carrying"]      = 0.65*pool['Dribbles per 90'] + 0.35*pool['Progressive runs per 90']
    pool["Aerial Requirement"] = pool['Aerial duels per 90'] * pool['Aerial duels won, %'] / 100.0
    pool["Passing Volume"]     = pool['Passes per 90']
    pool["Goal Output"]        = pool['Non-penalty goals per 90']
    pool["Retention"]          = pool['Accurate passes, %']

    cols = ["Opportunities","Ball Carrying","Aerial Requirement","Passing Volume","Goal Output","Retention"]
    for c in cols: pool[f"__tmpl__{c}"] = tmpl_vec[c]
    pool["BaseDist"] = pool.apply(lambda r: norm([r[c]-r[f"__tmpl__{c}"] for c in cols]), axis=1)

    ranked = _score_block(pool.copy())
    return ranked, pool, "Strikers (CF)", tmpl_src

def compute_attackers():
    # -------- Position filter (attacker) --------
    role_choice = st.radio(
        "Position filter (Attackers)",
        ["All", "Left Wingers", "Right Wingers", "Attacking Midfielders"],
        index=0,
        horizontal=True,
        key="att_role_choice",
    )

    def _primary_token(pos: str) -> str:
        p = str(pos).upper().strip()
        tokens = [t for t in re.split(r"[,/;]\s*|\s+", p) if t]
        return tokens[0] if tokens else ""

    def position_filter(pos: str) -> bool:
        t0 = _primary_token(pos)
        if role_choice == "All":
            allowed = {"RW", "RWF", "RAMF", "LW", "LWF", "LAMF", "AMF"}
            return t0 in allowed
        if role_choice == "Right Wingers":
            return t0 in {"RW", "RWF", "RAMF"}
        if role_choice == "Left Wingers":
            return t0 in {"LW", "LWF", "LAMF"}
        if role_choice == "Attacking Midfielders":
            return t0 == "AMF"
        return False

    # features for attackers (your snippet)
    feats = [
        'Accurate passes, %','xG per 90','Non-penalty goals per 90','Touches in box per 90',
        'xA per 90','Passes to penalty area per 90','Passes per 90',
        'Progressive passes per 90','Passes to final third per 90',
        'Dribbles per 90','Progressive runs per 90'
    ]

    tmpl_src = _template_rows_for_role(position_filter).dropna(subset=feats)
    if use_single_template_player:
        players = sorted(tmpl_src["Player"].dropna().astype(str).unique())
        chosen = st.selectbox("Template player (Attackers)", ["— Select —"] + players, index=0, key="att_tmpl_pick")
        if chosen and not chosen.startswith("—"):
            tmpl_src = tmpl_src[tmpl_src["Player"].astype(str) == chosen]
    if tmpl_src.empty:
        st.error("No attackers found for template conditions."); st.stop()

    f = tmpl_src.copy()
    f["Retention Style"]   = f['Accurate passes, %']
    f["Goal Threat"]       = 0.4*f['xG per 90'] + 0.4*f['Non-penalty goals per 90'] + 0.2*f['Touches in box per 90']
    f["Creativity Threat"] = 0.65*f['xA per 90'] + 0.35*f['Passes to penalty area per 90']
    f["Passing Volume"]    = f['Passes per 90']
    f["Deeper Playmaking"] = 0.5*f['Progressive passes per 90'] + 0.5*f['Passes to final third per 90']
    f["Ball Carrying"]     = 0.6*f['Dribbles per 90'] + 0.4*f['Progressive runs per 90']
    cols = ["Retention Style","Goal Threat","Creativity Threat","Passing Volume","Deeper Playmaking","Ball Carrying"]
    tmpl_vec = f[cols].mean()

    base_pool = build_base_pool()
    pool = base_pool[base_pool["Position"].apply(position_filter)].copy()
    pool = pool[~((pool["Team"].astype(str) == template_team) & (pool["League"].astype(str) == template_league))]
    # caps for attackers (your snippet)
    pool = pool[(pd.to_numeric(pool["Age"], errors="coerce") <= 23) &
                (pd.to_numeric(pool["Market value"], errors="coerce") <= 5_000_000) &
                (pd.to_numeric(pool["Minutes played"], errors="coerce") >= 900)]
    for c in feats: pool[c] = pd.to_numeric(pool[c], errors="coerce")
    pool = pool.dropna(subset=feats)

    pool["Retention Style"]   = pool['Accurate passes, %']
    pool["Goal Threat"]       = 0.4*pool['xG per 90'] + 0.4*pool['Non-penalty goals per 90'] + 0.2*pool['Touches in box per 90']
    pool["Creativity Threat"] = 0.65*pool['xA per 90'] + 0.35*pool['Passes to penalty area per 90']
    pool["Passing Volume"]    = pool['Passes per 90']
    pool["Deeper Playmaking"] = 0.5*pool['Progressive passes per 90'] + 0.5*pool['Passes to final third per 90']
    pool["Ball Carrying"]     = 0.6*pool['Dribbles per 90'] + 0.4*pool['Progressive runs per 90']

    for c in cols: pool[f"__tmpl__{c}"] = tmpl_vec[c]
    pool["BaseDist"] = pool.apply(lambda r: norm([r[c]-r[f"__tmpl__{c}"] for c in cols]), axis=1)

    ranked = _score_block(pool.copy())
    subset = {
        "All": "Wingers/AM",
        "Left Wingers": "Left Wingers",
        "Right Wingers": "Right Wingers",
        "Attacking Midfielders": "AMF",
    }[role_choice]
    return ranked, pool, f"Attackers ({subset})", tmpl_src

def compute_central_mid():
    feats = [
        'Passes per 90','Forward passes per 90',
        'Progressive passes per 90','Progressive runs per 90',
        'Defensive duels per 90','PAdj Interceptions',
        'Touches in box per 90','Shots per 90','Accurate passes, %'
    ]
    def pos_ok(p):
        s = str(p).strip().upper()
        return s.startswith(("DMF","CMF","LCMF","RCMF","LDMF","RDMF"))

    tmpl_src = _template_rows_for_role(pos_ok).dropna(subset=feats)
    if use_single_template_player:
        players = sorted(tmpl_src["Player"].dropna().astype(str).unique())
        chosen = st.selectbox("Template player (Central Midfield)", ["— Select —"] + players, index=0, key="cm_tmpl_pick")
        if chosen and not chosen.startswith("—"):
            tmpl_src = tmpl_src[tmpl_src["Player"].astype(str) == chosen]
    if tmpl_src.empty:
        st.error("No central midfielders found for template conditions."); st.stop()

    f = tmpl_src.copy()
    f["Pass Verticality"]    = f['Forward passes per 90'] / f['Passes per 90']
    f["Progression Volume"]  = f['Progressive passes per 90'] + f['Progressive runs per 90']
    f["Attacking Contribution"] = f['Touches in box per 90'] + f['Shots per 90']
    f["Defensive Volume"]    = f['Defensive duels per 90']
    f["Interception Volume"] = f['PAdj Interceptions']
    f["Retention"]           = f['Accurate passes, %']
    cols = ["Passes per 90","Pass Verticality","Progression Volume","Defensive Volume","Interception Volume","Attacking Contribution","Retention"]
    tmpl_vec = f[cols].mean()

    base_pool = build_base_pool()
    pool = base_pool[base_pool["Position"].apply(pos_ok)].copy()
    pool = pool[~((pool["Team"].astype(str) == template_team) & (pool["League"].astype(str) == template_league))]
    pool = pool[(pd.to_numeric(pool["Age"], errors="coerce") <= 32) &
                (pd.to_numeric(pool["Market value"], errors="coerce") <= 5_000_000) &
                (pd.to_numeric(pool["Minutes played"], errors="coerce") >= 1000)]
    for c in feats: pool[c] = pd.to_numeric(pool[c], errors="coerce")
    pool = pool.dropna(subset=feats)

    pool["Pass Verticality"]    = pool['Forward passes per 90'] / pool['Passes per 90']
    pool["Progression Volume"]  = pool['Progressive passes per 90'] + pool['Progressive runs per 90']
    pool["Attacking Contribution"] = pool['Touches in box per 90'] + pool['Shots per 90']
    pool["Defensive Volume"]    = pool['Defensive duels per 90']
    pool["Interception Volume"] = pool['PAdj Interceptions']
    pool["Retention"]           = pool['Accurate passes, %']

    for c in cols: pool[f"__tmpl__{c}"] = tmpl_vec[c]
    pool["BaseDist"] = pool.apply(lambda r: norm([r[c]-r[f"__tmpl__{c}"] for c in cols]), axis=1)

    ranked = _score_block(pool.copy())
    return ranked, pool, "Central Midfield", tmpl_src

def compute_fullbacks():
    # -------- Position filter (fullbacks) --------
    role_choice = st.radio(
        "Position filter (Fullbacks)",
        ["All", "Left Backs", "Right Backs"],
        index=0,
        horizontal=True,
        key="fb_role_choice",
    )

    feats = [
        'Passes per 90','Forward passes per 90',
        'Progressive passes per 90','Progressive runs per 90',
        'Defensive duels per 90','PAdj Interceptions','Aerial duels per 90',
        'xA per 90','Crosses per 90','Touches in box per 90',
        'Shots per 90','Passes to penalty area per 90',
        'Accurate passes, %'
    ]
    def position_filter(pos: str) -> bool:
        s = str(pos).strip().upper()
        tokens = [t for t in re.split(r"[,/;]\s*|\s+", s) if t]
        t0 = tokens[0] if tokens else ""
        if role_choice == "Right Backs":
            prefixes = ("RB", "RWB")
        elif role_choice == "Left Backs":
            prefixes = ("LB", "LWB")
        else:
            prefixes = ("RB", "RWB", "LB", "LWB")
        return any(t0.startswith(pf) for pf in prefixes)

    tmpl_src = _template_rows_for_role(position_filter).dropna(subset=feats)
    if use_single_template_player:
        players = sorted(tmpl_src["Player"].dropna().astype(str).unique())
        chosen = st.selectbox("Template player (Fullbacks)", ["— Select —"] + players, index=0, key="fb_tmpl_pick")
        if chosen and not chosen.startswith("—"):
            tmpl_src = tmpl_src[tmpl_src["Player"].astype(str) == chosen]
    if tmpl_src.empty:
        st.error("No fullbacks found for template conditions."); st.stop()

    f = tmpl_src.copy()
    f["Pass Verticality"]     = f['Forward passes per 90'] / f['Passes per 90']
    f["Progression Volume"]   = f['Progressive passes per 90'] + f['Progressive runs per 90']
    f["Attacking Contribution"]= 0.4*f['xA per 90'] + 0.2*f['Crosses per 90'] + 0.2*f['Touches in box per 90'] + 0.1*f['Shots per 90'] + 0.1*f['Passes to penalty area per 90']
    f["Defensive Volume"]     = 0.5*f['Defensive duels per 90'] + 0.3*f['PAdj Interceptions'] + 0.2*f['Aerial duels per 90']
    f["Retention"]            = f['Accurate passes, %']
    cols = ["Passes per 90","Pass Verticality","Progression Volume","Attacking Contribution","Defensive Volume","Retention"]
    tmpl_vec = f[cols].mean()

    base_pool = build_base_pool()
    pool = base_pool[base_pool["Position"].apply(position_filter)].copy()
    pool = pool[~((pool["Team"].astype(str) == template_team) & (pool["League"].astype(str) == template_league))]
    pool = pool[(pd.to_numeric(pool["Age"], errors="coerce") <= 30) &
                (pd.to_numeric(pool["Market value"], errors="coerce") <= 10_000_000) &
                (pd.to_numeric(pool["Minutes played"], errors="coerce") >= 1000)]
    for c in feats: pool[c] = pd.to_numeric(pool[c], errors="coerce")
    pool = pool.dropna(subset=feats)

    pool["Pass Verticality"]     = pool['Forward passes per 90'] / pool['Passes per 90']
    pool["Progression Volume"]   = pool['Progressive passes per 90'] + pool['Progressive runs per 90']
    pool["Attacking Contribution"]= 0.4*pool['xA per 90'] + 0.2*pool['Crosses per 90'] + 0.2*pool['Touches in box per 90'] + 0.1*pool['Shots per 90'] + 0.1*pool['Passes to penalty area per 90']
    pool["Defensive Volume"]     = 0.5*pool['Defensive duels per 90'] + 0.3*pool['PAdj Interceptions'] + 0.2*pool['Aerial duels per 90']
    pool["Retention"]            = pool['Accurate passes, %']

    for c in cols: pool[f"__tmpl__{c}"] = tmpl_vec[c]
    pool["BaseDist"] = pool.apply(lambda r: norm([r[c]-r[f"__tmpl__{c}"] for c in cols]), axis=1)

    ranked = _score_block(pool.copy())
    subset = {
        "All": "Fullbacks",
        "Left Backs": "Left Backs",
        "Right Backs": "Right Backs",
    }[role_choice]
    return ranked, pool, subset, tmpl_src

def compute_center_backs():
    feats = [
        'Aerial duels per 90','Defensive duels per 90',
        'Passes per 90','Forward passes per 90',
        'Progressive passes per 90','Progressive runs per 90',
        'PAdj Interceptions','Shots blocked per 90'
    ]
    def pos_ok(p):
        s = str(p).strip().upper()
        return s.startswith(("CB","RCB","LCB"))

    tmpl_src = _template_rows_for_role(pos_ok).dropna(subset=feats)
    if use_single_template_player:
        players = sorted(tmpl_src["Player"].dropna().astype(str).unique())
        chosen = st.selectbox("Template player (Center Backs)", ["— Select —"] + players, index=0, key="cb_tmpl_pick")
        if chosen and not chosen.startswith("—"):
            tmpl_src = tmpl_src[tmpl_src["Player"].astype(str) == chosen]
    if tmpl_src.empty:
        st.error("No centre-backs found for template conditions."); st.stop()

    f = tmpl_src.copy()
    f["Passing Verticality"] = f['Forward passes per 90'] / f['Passes per 90']
    f["Passing Volume"]      = f['Passes per 90']
    f["Positional Demand"]   = f['PAdj Interceptions'] + f['Shots blocked per 90']
    f["Progression Volume"]  = f['Progressive passes per 90'] + f['Progressive runs per 90']
    cols = ["Aerial duels per 90","Defensive duels per 90","Positional Demand","Passing Volume","Passing Verticality","Progression Volume"]
    tmpl_vec = f[cols].mean()

    base_pool = build_base_pool()
    pool = base_pool[base_pool["Position"].apply(pos_ok)].copy()
    pool = pool[~((pool["Team"].astype(str) == template_team) & (pool["League"].astype(str) == template_league))]
    # your CB caps
    pool = pool[(pd.to_numeric(pool["Age"], errors="coerce") <= 22) &
                (pd.to_numeric(pool["Market value"], errors="coerce") <= 10_000_000) &
                (pd.to_numeric(pool["Minutes played"], errors="coerce") >= 500)]
    for c in feats: pool[c] = pd.to_numeric(pool[c], errors="coerce")
    pool = pool.dropna(subset=feats)

    pool["Passing Verticality"] = pool['Forward passes per 90'] / pool['Passes per 90']
    pool["Passing Volume"]      = pool['Passes per 90']
    pool["Positional Demand"]   = pool['PAdj Interceptions'] + pool['Shots blocked per 90']
    pool["Progression Volume"]  = pool['Progressive passes per 90'] + pool['Progressive runs per 90']

    for c in cols: pool[f"__tmpl__{c}"] = tmpl_vec[c]
    pool["BaseDist"] = pool.apply(lambda r: norm([r[c]-r[f"__tmpl__{c}"] for c in cols]), axis=1)

    ranked = _score_block(pool.copy())
    return ranked, pool, "Center Backs", tmpl_src

# ======================== UI: tabs for roles ========================
tab_st, tab_att, tab_cm, tab_fb, tab_cb = st.tabs(
    ["Strikers", "Attackers", "Central Midfield", "Fullbacks", "Center Backs"]
)

# We'll render tiles + Feature-Z inside each tab via a shared renderer
# ---------- Style for tiles ----------
st.markdown(
    """
<style>
  :root { --bg:#0f1115; --card:#161a22; --muted:#a8b3cf; --soft:#202633; }
  .block-container { padding-top:.8rem; }
  body{ background:var(--bg); font-family: system-ui,-apple-system,'Segoe UI','Segoe UI Emoji',Roboto,Helvetica,Arial,sans-serif;}
  .wrap{ display:flex; justify-content:center; }
  .player-card{
    width:min(980px,96%); display:grid; grid-template-columns:112px 1fr 100px;
    gap:14px; align-items:start; background:var(--card); border:1px solid #252b3a;
    border-radius:18px; padding:16px; box-shadow: 0 2px 14px rgba(0,0,0,.25);
  }
  .avatar{
    width:112px; height:112px; border-radius:12px;
    background-color:#0b0d12; background-size:cover; background-position:center;
    border:1px solid #2a3145;
  }
  .leftcol{ display:flex; flex-direction:column; align-items:center; gap:8px; }
  .name{ font-weight:800; font-size:22px; color:#e8ecff; margin-bottom:6px; }
  .sub{ color:#a8b3cf; font-size:15px; }
  .pill{ padding:2px 10px; border-radius:9px; font-weight:800; font-size:18px; color:#0b0d12; display:inline-block; min-width:42px; text-align:center; }
  .row{ display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin:4px 0; }
  .chip{ background:var(--soft); color:#cbd5f5; border:1px solid #2d3550; padding:3px 10px; border-radius:10px; font-size:13px; line-height:18px; }
  .pos{ color:#eaf0ff; font-weight:700; padding:4px 10px; border-radius:10px; font-size:12px; border:1px solid rgba(255,255,255,.08); }
  .teamline{ color:#e6ebff; font-size:15px; font-weight:400; margin-top:2px; }
  .fit{ color:#94f0c8; font-weight:900; font-size:28px; text-align:right; }
  .fit small{ display:block; color:#9fb3c6; font-weight:600; font-size:12px; margin-top:4px; }
  .divider{ height:12px; }
  .metric-section{ background:#121621; border:1px solid #242b3b; border-radius:14px; padding:10px 12px; }
  .m-title{ color:#e8ecff; font-weight:800; letter-spacing:.02em; margin:4px 0 10px 0; font-size:20px; text-transform:uppercase; }
  .m-row{ display:flex; justify-content:space-between; align-items:center; padding:8px 8px; border-radius:10px; }
  .m-row + .m-row{ margin-top:6px; }
  .m-label{ color:#c9d3f2; font-size:16px; }
  .m-right{ display:flex; align-items:center; gap:8px; }
  .m-badge{ min-width:40px; text-align:center; padding:2px 10px; border-radius:8px; font-weight:800; font-size:18px; color:#0b0d12; border:1px solid rgba(0,0,0,.15); }
  .metrics-grid{ display:grid; grid-template-columns:1fr; gap:12px; }
  @media (min-width: 980px){ .metrics-grid{ grid-template-columns:repeat(3, 1fr); } }
</style>
""",
    unsafe_allow_html=True,
)

PALETTE=[(0,(208,2,27)),(50,(245,166,35)),(65,(248,231,28)),(75,(126,211,33)),(85,(65,117,5)),(100,(40,90,4))]
def _lerp(a,b,t): return tuple(int(round(a[i]+(b[i]-a[i])*t)) for i in range(3))
def rating_color(v:float)->str:
    v=max(0.0,min(100.0,float(v)))
    for i in range(len(PALETTE)-1):
        x0,c0=PALETTE[i]; x1,c1=PALETTE[i+1]
        if v<=x1:
            t=0 if x1==x0 else (v-x0)/(x1-x0); r,g,b=_lerp(c0,c1,t); return f"rgb({r},{g},{b})"
    r,g,b=PALETTE[-1][1]; return f"rgb({r},{g},{b})"

# ====================== PlaymakerStats image resolver ======================
_PS_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.playmakerstats.com/",
}

@st.cache_data(show_spinner=False, ttl=24*3600)
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

def _extract_og_image(html: str):
    m = re.search(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']', html, flags=re.I)
    return m.group(1) if m else None

@st.cache_data(show_spinner=False, ttl=24*3600)
def playmakerstats_image_by_name_team(name: str, team: str):
    q = f"{name} {team}".strip()
    # PlaymakerStats (EN)
    search_url = f"https://www.playmakerstats.com/search.php?search={quote(q)}"
    html = _http_get_text(search_url, retries=1)
    if html:
        links = re.findall(r'href=["\'](/(?:player|jogador)\.php\?id=\d+[^"\']*)', html, flags=re.I)
        if links:
            p_html = _http_get_text("https://www.playmakerstats.com" + links[0], retries=1)
            if p_html:
                img = _extract_og_image(p_html)
                if img:
                    return img
    # zerozero.pt fallback
    search_url2 = f"https://www.zerozero.pt/procura.php?search={quote(q)}"
    html2 = _http_get_text(search_url2, retries=1)
    if html2:
        links2 = re.findall(r'href=["\'](/jogador\.php\?id=\d+[^"\']*)', html2, flags=re.I)
        if links2:
            p_html2 = _http_get_text("https://www.zerozero.pt" + links2[0], retries=1)
            if p_html2:
                img2 = _extract_og_image(p_html2)
                if img2:
                    return img2
    return None

PLACEHOLDER_IMG = "https://i.redd.it/43axcjdu59nd1.jpeg"
if "photo_map" not in st.session_state:
    st.session_state["photo_map"] = {}

# ---------- shared tile+FeatureZ renderer ----------
def render_tiles_and_featureZ(ranked: pd.DataFrame, df_pool_role: pd.DataFrame, role_title: str):
    st.markdown("---")
    st.header(f"🏅 Top Role Matches — Tiles · {role_title}")

    # Helpers for dropdown metrics (percentiles vs pool)
    def pct_series_for_player(player_row: pd.Series, col: str, within_df: pd.DataFrame) -> float:
        vals = pd.to_numeric(within_df[col], errors="coerce").dropna()
        if vals.empty: return 0.0
        v = pd.to_numeric(player_row.get(col), errors="coerce")
        if pd.isna(v): return 0.0
        return float((vals <= v).mean() * 100.0)

    st.write("")  # tiny spacer
    for idx, row in ranked.head(int(top_n)).iterrows():
        name  = str(row.get("Player",""))
        team  = str(row.get("Team",""))
        league= str(row.get("League",""))
        pos   = str(row.get("Position",""))
        age   = int(row.get("Age",0)) if not pd.isna(row.get("Age",np.nan)) else 0
        minutes = int(row.get("Minutes played",0)) if not pd.isna(row.get("Minutes played",np.nan)) else 0
        fit   = float(row.get("Role Fit Score",0.0))
        fit_pct = max(0, min(100, int(round(fit))))

        key_id = f"{name}|||{team}|||{league}"
        avatar_url = playmakerstats_image_by_name_team(name, team) or PLACEHOLDER_IMG
        override_url = st.session_state.get("photo_map", {}).get(key_id, "")
        if override_url:
            avatar_url = override_url + f"?t={int(time.time())}"
        if DEBUG_PHOTOS:
            st.write(f"PHOTO DEBUG → '{name}' / '{team}' → {avatar_url}")

        ov_style = f"background:{rating_color(fit_pct)};"

        codes = [c for c in re.split(r"[,/; ]+", pos.strip().upper()) if c]
        chips_html = " ".join(f"<span class='pos'>{c}</span>" for c in dict.fromkeys(codes))

        st.markdown(f"""
        <div class='wrap'>
          <div class='player-card'>
            <div class='leftcol'>
              <div class='avatar' style="background-image:url('{avatar_url}');"></div>
              <div class='row'><span class='chip'>{age}y</span><span class='chip'>{minutes}m</span></div>
            </div>
            <div>
              <div class='name'>{name}</div>
              <div class='row' style='align-items:center;'>
                <span class='pill' style='{ov_style}'>{fit_pct}</span>
                <span class='sub'>Overall Fit</span>
              </div>
              <div class='row'>{chips_html}</div>
              <div class='teamline'>{team} · {league}</div>
            </div>
            <div class='fit'>{fit_pct}%<small>Fit</small></div>
          </div>
        </div>
        <div class='divider'></div>
        """, unsafe_allow_html=True)

        # === dropdown: metrics + image URL override ===
        with st.expander("▼ Show individual metrics / Set photo override"):
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
                if met in df_pool_role.columns:
                    ATTACKING.append((lab, pct_series_for_player(row, met, df_pool_role)))

            DEFENSIVE = []
            for lab, met in [
                ("Aerial Duels","Aerial duels per 90"),
                ("Aerial Duel Success %","Aerial duels won, %"),
                ("PAdj. Interceptions","PAdj Interceptions"),
                ("Defensive Duels","Defensive duels per 90"),
                ("Defensive Duel Success %","Defensive duels won, %"),
            ]:
                if met in df_pool_role.columns:
                    DEFENSIVE.append((lab, pct_series_for_player(row, met, df_pool_role)))

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
                if met in df_pool_role.columns:
                    POSSESSION.append((lab, pct_series_for_player(row, met, df_pool_role)))

            def section_html(title: str, items):
                rows=[]
                for lab, pct in items:
                    pct_i = int(round(max(0.0, min(100.0, float(pct)))))
                    rows.append(
                        f"<div class='m-row'><div class='m-label'>{lab}</div>"
                        f"<div class='m-right'><span class='m-badge' style='background:{rating_color(pct_i)}'>{pct_i}</span></div></div>"
                    )
                return f"<div class='metric-section'><div class='m-title'>{title}</div>{''.join(rows)}</div>"

            col_html = (
                "<div class='metrics-grid'>"
                + section_html('ATTACKING', ATTACKING)
                + section_html('DEFENSIVE', DEFENSIVE)
                + section_html('POSSESSION', POSSESSION)
                + "</div>"
            )
            st.markdown(col_html, unsafe_allow_html=True)

            # --- Custom image URL override ---
            img_key = f"imgurl_{key_id}"
            default_url = st.session_state.get("photo_map", {}).get(key_id, "")
            _ = st.text_input(
                "Custom image URL (override avatar — e.g., https://images.fotmob.com/image_resources/playerimages/1199383.png)",
                value=default_url,
                key=img_key
            )

            col_a, col_b = st.columns([1, 3])
            with col_a:
                if st.button("Apply to this player", key=f"apply_{key_id}"):
                    val = (st.session_state.get(img_key, "") or "").strip()
                    if not val:
                        st.error("Please paste an image URL.")
                    elif not (val.startswith("http://") or val.startswith("https://")):
                        st.error("Image URL must start with http:// or https://")
                    else:
                        st.session_state.setdefault("photo_map", {})[key_id] = val
                        st.success("Saved!")
                        try: st.rerun()
                        except Exception: st.experimental_rerun()

            with col_b:
                if st.button("Clear override", key=f"clear_{key_id}"):
                    st.session_state["photo_map"].pop(key_id, None)
                    st.info("Cleared.")
                    try: st.rerun()
                    except Exception: st.experimental_rerun()

    # ======================== Feature Z (unchanged, but scoped to this role’s df_pool) ========================
    st.markdown("---")
    st.header(f"Advanced Individual Player Analysis (Feature Z) · {role_title}")

    import matplotlib.pyplot as plt
    from matplotlib.transforms import ScaledTranslation
    from matplotlib.font_manager import FontProperties
    from PIL import Image  # noqa: F401

    left, right = st.columns([2,2])
    with left:
        options_ranked = ranked["Player"].astype(str).head(int(top_n)).tolist()
        any_pool = st.checkbox("Pick from entire candidate pool (not just Top N)", value=False, key=f"fz_pool_toggle_{role_title}")
        options = sorted(df_pool_role["Player"].dropna().astype(str).unique()) if any_pool else options_ranked
        if not options:
            st.info("No players available for Feature Z. Adjust filters."); return
        player_sel = st.selectbox("Choose player for Feature Z", options, index=0, key=f"fz_pick_{role_title}")

    with right:
        show_height = st.checkbox("Show height in info row", value=True, key=f"fz_height_{role_title}")
        foot_override_on = st.checkbox("Edit foot value", value=False, key=f"fz_foot_on_{role_title}")
        foot_override_text = st.text_input("Foot (e.g., Left)", value="", disabled=not foot_override_on, key=f"fz_foot_txt_{role_title}")
        name_override_on = st.checkbox("Edit display name", value=False, key=f"fz_name_on_{role_title}")
        name_override = st.text_input("Display name", "", disabled=not name_override_on, key=f"fz_name_txt_{role_title}")
        footer_caption_text = st.text_input("Footer caption", "Percentile Rank", key=f"fz_footer_{role_title}")

    player_row = df_pool_role[df_pool_role["Player"].astype(str) == str(player_sel)].head(1)
    if player_row.empty:
        st.info("Pick a player above."); return

    def _safe_get(sr, key, default="—"):
        try:
            v = sr.iloc[0].get(key, default)
            s = "" if v is None else str(v)
            return default if s.strip() == "" else s
        except Exception:
            return default

    def pct_series(col: str) -> float:
        vals = pd.to_numeric(df_pool_role[col], errors="coerce").dropna()
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

    pos   = _safe_get(player_row, "Position", "—")
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
            height_text = str(v).strip(); break

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
        if met in df_pool_role.columns:
            ATTACKING.append((lab, float(np.nan_to_num(pct_series(met), nan=0.0)), val_of(met)[1]))

    for lab, met in [
        ("Aerial Duels","Aerial duels per 90"),
        ("Aerial Duel Success %","Aerial duels won, %"),
        ("PAdj. Interceptions","PAdj Interceptions"),
        ("Defensive Duels","Defensive duels per 90"),
        ("Defensive Duel Success %","Defensive duels won, %"),
    ]:
        if met in df_pool_role.columns:
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
        if met in df_pool_role.columns:
            POSSESSION.append((lab, float(np.nan_to_num(pct_series(met), nan=0.0)), val_of(met)[1]))

    sections = [("Attacking",ATTACKING),("Defensive",DEFENSIVE),("Possession",POSSESSION)]
    sections = [(t,lst) for t,lst in sections if lst]

    def _font_name_or_fallback(pref, fallback="DejaVu Sans"):
        from matplotlib import font_manager as fm
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

    TAB_RED=np.array([199,54,60]); TAB_GOLD=np.array([240,197,106]); TAB_GREEN=np.array([61,166,91])
    def pct_to_rgb(v):
        v=float(np.clip(v,0,100))
        def _blend(c1,c2,t): c=c1+(c2-c1)*np.clip(t,0,1); return f"#{int(c[0]):02x}{int(c[1]):02x}{int(c[2]):02x}"
        return _blend(TAB_RED,TAB_GOLD,v/50) if v<=50 else _blend(TAB_GOLD,TAB_GREEN,(v-50)/50)

    fig_size   = (11.8, 9.6); dpi = 120
    title_row_h = 0.125; header_block_h = title_row_h + 0.055
    fig = plt.figure(figsize=fig_size, dpi=dpi); fig.patch.set_facecolor(PAGE_BG)
    try:
        # Ensure renderer is ready before measuring text extents
        fig.canvas.draw()
    except Exception:
        pass

    fig.text(LEFT, 1 - TOP - 0.010, f"{name_}\u2009|\u2009{team}",
            ha="left", va="top", color=TITLE_C, fontproperties=TITLE_FP)

    def draw_pairs_line(pairs_line, y):
        x = LEFT
        try:
            renderer = fig.canvas.get_renderer()
        except Exception:
            renderer = None
        for i,(lab,val) in enumerate(pairs_line):
            t1 = fig.text(x, y, lab, ha="left", va="top", color=LABEL_C, fontproperties=INFO_LABEL_FP)
            if renderer is not None:
                fig.canvas.draw()
                x += t1.get_window_extent(renderer).width / fig.bbox.width
            else:
                x += 0.06
            t2 = fig.text(x, y, str(val), ha="left", va="top", color=LABEL_C, fontproperties=INFO_VALUE_FP)
            if renderer is not None:
                fig.canvas.draw(); x += t2.get_window_extent(renderer).width / fig.bbox.width
            else:
                x += 0.06
            if i != len(pairs_line)-1:
                t3 = fig.text(x, y, "  |  ", ha="left", va="top", color="#555555", fontproperties=INFO_VALUE_FP)
                if renderer is not None:
                    fig.canvas.draw(); x += t3.get_window_extent(renderer).width / fig.bbox.width
                else:
                    x += 0.03

    row1 = [("Position: ",pos), ("Age: ",age), ("Height: ", height_text if (show_height and height_text) else "—")]
    row2 = [("Games: ",games), ("Goals: ",goals), ("Assists: ",assists)]
    row3 = [("Minutes: ",minutes), ("Foot: ",foot_display)]
    title_y = 1 - TOP - 0.010
    y1 = title_y - 0.055; y2 = y1 - 0.039; y3 = y2 - 0.039
    draw_pairs_line(row1, y1); draw_pairs_line(row2, y2); draw_pairs_line(row3, y3)

    fig.lines.append(plt.Line2D([LEFT, 1 - RIGHT],[1 - TOP - header_block_h + 0.004]*2,
                                transform=fig.transFigure, color=DIVIDER, lw=0.8, alpha=0.35))

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

        for i, (lab, _, _) in enumerate(tuples[::-1]):
            y_fig = (panel_top - header_h - n * row_slot) + ((i + 0.5) * row_slot)
            fig.text(LEFT, y_fig, lab, ha="left", va="center", color=LABEL_C, fontproperties=LABEL_FP)

        if show_xticks:
            trans = ax.get_xaxis_transform()
            offset_inner   = ScaledTranslation(7/72, 0, fig.dpi_scale_trans)
            offset_pct_0   = ScaledTranslation(4/72, 0, fig.dpi_scale_trans)
            offset_pct_100 = ScaledTranslation(10/72, 0, fig.dpi_scale_trans)
            y_label = -0.075
            for gx in ticks:
                ax.plot([gx, gx], [-0.03, 0.0], transform=trans, color=(0, 0, 0, 0.6), lw=1.1, clip_on=False, zorder=4)
                ax.text(gx, y_label, f"{int(gx)}", transform=trans, ha="center", va="top",
                        color="#000", fontproperties=TICK_FP, zorder=4, clip_on=False)
                if gx == 0:
                    ax.text(gx, y_label, "%", transform=trans + offset_pct_0, ha="left", va="top",
                            color="#000", fontproperties=TICK_FP)
                elif gx == 100:
                    ax.text(gx, y_label, "%", transform=trans + offset_pct_100, ha="left", va="top",
                            color="#000", fontproperties=TICK_FP)
                else:
                    ax.text(gx, y_label, "%", transform=trans + offset_inner, ha="left", va="top",
                            color="#000", fontproperties=TICK_FP)

        if draw_bottom_divider:
            y0 = (panel_top - header_h - n * row_slot) - 0.008
            fig.lines.append(plt.Line2D([LEFT, 1 - RIGHT], [y0, y0], transform=fig.transFigure,
                                        color=DIVIDER, lw=1.2, alpha=0.35))
        return (panel_top - header_h - n * row_slot) - GAP

    y_top = 1 - TOP - header_block_h
    for sec_idx, (sec_title, sec_data) in enumerate(sections):
        last = (sec_idx == len(sections) - 1)
        y_top = draw_panel(y_top, sec_title, sec_data, show_xticks=last, draw_bottom_divider=not last)

    fig.text((LEFT + gutter + (1 - RIGHT)) / 2.0, BOT * 0.1,
            footer_caption_text if 'footer_caption_text' in locals() else "Percentile Rank",
            ha="center", va="center", color="#222222", fontproperties=FOOTER_FP)

    st.pyplot(fig, use_container_width=True)

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

# ======================== per-tab compute + render ========================
with tab_st:
    ranked, pool, tag, tmpl_src = compute_strikers()
    st.subheader("🧩 Players used for Striker Role Template")
    showcols = [c for c in ["Player","Minutes played","Position","League"] if c in tmpl_src.columns]
    if showcols:
        st.dataframe(tmpl_src[showcols].sort_values("Minutes played", ascending=False), use_container_width=True)
    render_tiles_and_featureZ(ranked, pool, tag)

with tab_att:
    ranked, pool, tag, tmpl_src = compute_attackers()
    st.subheader("🧩 Players used for Attacker Role Template")
    showcols = [c for c in ["Player","Minutes played","Position","League"] if c in tmpl_src.columns]
    if showcols:
        st.dataframe(tmpl_src[showcols].sort_values("Minutes played", ascending=False), use_container_width=True)
    render_tiles_and_featureZ(ranked, pool, tag)

with tab_cm:
    ranked, pool, tag, tmpl_src = compute_central_mid()
    st.subheader("🧩 Players used for CM Role Template")
    showcols = [c for c in ["Player","Minutes played","Position","League"] if c in tmpl_src.columns]
    if showcols:
        st.dataframe(tmpl_src[showcols].sort_values("Minutes played", ascending=False), use_container_width=True)
    render_tiles_and_featureZ(ranked, pool, tag)

with tab_fb:
    ranked, pool, tag, tmpl_src = compute_fullbacks()
    st.subheader("🧩 Players used for FB Role Template")
    showcols = [c for c in ["Player","Minutes played","Position","League"] if c in tmpl_src.columns]
    if showcols:
        st.dataframe(tmpl_src[showcols].sort_values("Minutes played", ascending=False), use_container_width=True)
    render_tiles_and_featureZ(ranked, pool, tag)

with tab_cb:
    ranked, pool, tag, tmpl_src = compute_center_backs()
    st.subheader("🧩 Players used for CB Role Template")
    showcols = [c for c in ["Player","Minutes played","Position","League"] if c in tmpl_src.columns]
    if showcols:
        st.dataframe(tmpl_src[showcols].sort_values("Minutes played", ascending=False), use_container_width=True)
    render_tiles_and_featureZ(ranked, pool, tag)












