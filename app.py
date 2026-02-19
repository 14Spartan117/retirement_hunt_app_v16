import io
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from finance_model import (
    ACCOUNTS,
    Phase,
    simulate_accounts,
    make_growth_figure,
    crossing_years,
    retirement_year_from_targets,
    build_report_pdf_bytes,
    simulate_yahoo_dca_monthly_values,
    solve_extra_monthly_contribution,
)

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"

DEFAULT_BASELINE_XLSX = DATA_DIR / "FY26_Army_Income_Budget_Investment_Baseline.xlsx"
DEFAULT_CSP_XLSX = DATA_DIR / "IWT_Conscious_Spending_Plan_2023.xlsx"

BUILD = "v16"


# -----------------------
# Pending widget-state updates (apply BEFORE widgets are instantiated)
# This avoids: StreamlitAPIException: cannot modify session_state after widget instantiation.
# -----------------------
if "_pending_state_updates" in st.session_state:
    _updates = st.session_state.pop("_pending_state_updates") or {}
    # Apply updates before widgets are created
    for _k, _v in _updates.items():
        st.session_state[_k] = _v

def queue_state_updates(updates: dict):
    """Queue widget-key updates to be applied on the next rerun (before widgets are created)."""
    if not isinstance(updates, dict):
        return
    # Merge if there are already pending updates
    prior = st.session_state.get("_pending_state_updates", {}) or {}
    prior.update(updates)
    st.session_state["_pending_state_updates"] = prior
    st.rerun()

st.set_page_config(page_title="Retirement Hunt", layout="wide")

# -----------------------
# Helpers
# -----------------------
def money(x, decimals=0):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return f"${float(x):,.{decimals}f}"

@st.cache_data(show_spinner=False)
def load_excel(path: str) -> dict:
    """Return dict of sheet_name -> DataFrame(header=None)."""
    xls = pd.ExcelFile(path)
    return {name: pd.read_excel(path, sheet_name=name, header=None) for name in xls.sheet_names}

def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="ignore")
    return out

def make_unique_index(labels):
    seen = {}
    out = []
    for x in labels:
        s = "" if (x is None or (isinstance(x, float) and np.isnan(x))) else str(x).strip()
        if s == "":
            out.append("")
            continue
        if s not in seen:
            seen[s] = 0
            out.append(s)
        else:
            seen[s] += 1
            out.append(f"{s} ({seen[s]+1})")
    return out

def parse_stacking_bread(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Expect first row: ['Year', 2024, 2025, ...]
    First col = metric labels.
    Returns DataFrame indexed by unique metric names, with year columns.
    """
    if df_raw.empty:
        return pd.DataFrame()

    top = df_raw.iloc[0].tolist()
    years = []
    col_idxs = []
    for j in range(1, len(top)):
        v = top[j]
        try:
            y = int(float(v))
            if 1900 <= y <= 2100:
                years.append(y)
                col_idxs.append(j)
        except Exception:
            pass

    if not years:
        return pd.DataFrame()

    body = df_raw.iloc[1:, [0] + col_idxs].copy()
    body.columns = ["Metric"] + [str(y) for y in years]
    body["Metric"] = make_unique_index(body["Metric"].tolist())
    body = body[body["Metric"] != ""].copy()

    # numeric
    for y in years:
        body[str(y)] = pd.to_numeric(body[str(y)], errors="coerce")

    out = body.set_index("Metric")
    return out

def parse_conscious_plan(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Expect two columns: item + value.
    Produces tidy table: Section, Item, Amount.
    """
    if df_raw.empty:
        return pd.DataFrame(columns=["Section","Item","Amount"])

    d = df_raw.copy()
    # Use first two columns only if more exist
    if d.shape[1] > 2:
        d = d.iloc[:, :2]
    d.columns = ["Item", "Amount"]

    d["Item"] = d["Item"].astype("object")
    # Normalize blank/NaN
    d["Item"] = d["Item"].where(d["Item"].notna(), "")
    d["Item"] = d["Item"].map(lambda x: str(x).strip())
    d["Amount"] = pd.to_numeric(d["Amount"], errors="coerce")

    # Identify headings (upper-ish, no amount)
    headings = set([
        "NET WORTH",
        "INCOME",
        "FIXED COSTS (50-60% of take home)",
        "INVESTMENTS (10% of take home)",
        "SAVINGS GOALS (5-10% of take home)",
        "GUILT-FREE SPENDING (20-35% of take home)",
    ])

    section = ""
    rows = []
    for _, r in d.iterrows():
        item = r["Item"]
        amt = r["Amount"]

        if item in headings:
            section = item
            continue

        if item == "" and (amt is None or (isinstance(amt, float) and np.isnan(amt))):
            continue

        # Skip totals rows — we will recompute them
        if item.upper().endswith("TOTAL") or "TOTAL NET WORTH" in item.upper():
            continue

        rows.append({"Section": section, "Item": item, "Amount": amt if not np.isnan(amt) else 0.0})

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["Section","Item","Amount"])
    return out

def apply_percent_to_items(df: pd.DataFrame, section: str, pct: float, include_zero=False) -> pd.DataFrame:
    """Apply +pct% to Amount for given section (or 'All')."""
    out = df.copy()
    mask = (out["Section"] == section) if section != "All" else np.ones(len(out), dtype=bool)
    if not include_zero:
        mask = mask & (out["Amount"] != 0)
    out.loc[mask, "Amount"] = out.loc[mask, "Amount"] * (1.0 + float(pct) / 100.0)
    return out

# -----------------------
# Caching for DCA runs
# -----------------------
@st.cache_data(show_spinner=False)
def cached_dca(
    ticker: str,
    start_date: date,
    N_years: int,
    monthly_contribs: tuple,
    buy_on: tuple,
    invest_initial_balance: float,
) -> dict:
    arr = np.array(monthly_contribs, dtype=float)
    return simulate_yahoo_dca_monthly_values(
        ticker=ticker,
        start_date=start_date,
        N_years=N_years,
        monthly_contribs=arr,
        buy_on=buy_on,
        price_col="Close",
        auto_adjust=True,
        invest_initial_balance=float(invest_initial_balance),
    )

# -----------------------
# App layout with tabs
# -----------------------
st.title("Retirement Hunt")
st.caption(f"Build: {BUILD}")

tabs = st.tabs(["Retirement Hunt", "FY26 Baseline (Excel)", "Conscious Spending Plan (Excel)"])

# ======================================================================================
# TAB 1: Retirement Hunt (main)
# ======================================================================================
with tabs[0]:
    st.caption(
        "Model IRA / 401K / 529 / ETF growth with phased contributions. "
        "Optionally backtest any account as a Yahoo Finance ticker (buy on 1st and/or 15th trading day)."
    )

    periods_per_year = 12

    # -----------------------
    # SIDEBAR: Inputs (use keys so other tabs can write into them)
    # -----------------------
    with st.sidebar:
        st.header("Starting Balances ($)")
        ira_s = st.number_input("IRA start", min_value=0.0, value=st.session_state.get("start_ira", 50_000.0),
                                step=1_000.0, format="%.2f", key="start_ira")
        k401_s = st.number_input("401K start", min_value=0.0, value=st.session_state.get("start_401k", 75_000.0),
                                 step=1_000.0, format="%.2f", key="start_401k")
        c529_s = st.number_input("529 start", min_value=0.0, value=st.session_state.get("start_529", 10_000.0),
                                 step=500.0, format="%.2f", key="start_529")
        etf_s = st.number_input("ETF start", min_value=0.0, value=st.session_state.get("start_etf", 25_000.0),
                                step=1_000.0, format="%.2f", key="start_etf")

        st.header("Annual Interest Rates (%) (used when Market Mode is OFF)")
        ira_i = st.number_input("IRA rate (%/yr)", min_value=0.0, value=st.session_state.get("rate_ira", 7.0),
                                step=0.1, format="%.2f", key="rate_ira")
        k401_i = st.number_input("401K rate (%/yr)", min_value=0.0, value=st.session_state.get("rate_401k", 7.0),
                                 step=0.1, format="%.2f", key="rate_401k")
        c529_i = st.number_input("529 rate (%/yr)", min_value=0.0, value=st.session_state.get("rate_529", 6.0),
                                 step=0.1, format="%.2f", key="rate_529")
        etf_i = st.number_input("ETF rate (%/yr)", min_value=0.0, value=st.session_state.get("rate_etf", 7.0),
                                step=0.1, format="%.2f", key="rate_etf")

        st.header("Simulation Horizon")
        horizon_years = st.number_input("Years to simulate", min_value=1, value=st.session_state.get("horizon_years", 40),
                                        step=1, key="horizon_years")

        st.header("Contribution Phases")
        num_phases = st.number_input("Number of phases", min_value=1, value=st.session_state.get("num_phases", 2),
                                     step=1, key="num_phases")

        phases = []
        for p in range(int(num_phases)):
            with st.expander(f"Phase {p+1}", expanded=(p == 0)):
                col1, col2 = st.columns(2)
                with col1:
                    start_year = st.number_input(
                        f"Start year (Phase {p+1})", min_value=0,
                        value=st.session_state.get(f"p{p}_sy", 0 if p == 0 else 10),
                        step=1, key=f"p{p}_sy",
                    )
                with col2:
                    end_year = st.number_input(
                        f"End year (Phase {p+1})", min_value=0,
                        value=st.session_state.get(f"p{p}_ey", 10 if p == 0 else int(horizon_years)),
                        step=1, key=f"p{p}_ey",
                    )

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    ira_c = st.number_input(
                        f"IRA / month (P{p+1})", min_value=0.0,
                        value=st.session_state.get(f"p{p}_ira", 500.0),
                        step=50.0, key=f"p{p}_ira",
                    )
                with c2:
                    tsp_c = st.number_input(
                        f"401K / month (P{p+1})", min_value=0.0,
                        value=st.session_state.get(f"p{p}_tsp", 1500.0),
                        step=50.0, key=f"p{p}_tsp",
                    )
                with c3:
                    col_c = st.number_input(
                        f"529 / month (P{p+1})", min_value=0.0,
                        value=st.session_state.get(f"p{p}_529", 250.0),
                        step=25.0, key=f"p{p}_529",
                    )
                with c4:
                    etf_c = st.number_input(
                        f"ETF / month (P{p+1})", min_value=0.0,
                        value=st.session_state.get(f"p{p}_etf", 500.0),
                        step=50.0, key=f"p{p}_etf",
                    )

                phases.append(
                    Phase(
                        phase=p + 1,
                        start_year=int(start_year),
                        end_year=int(end_year),
                        ira=float(ira_c),
                        tsp=float(tsp_c),
                        college=float(col_c),
                        etf=float(etf_c),
                    )
                )

        st.header("Market Mode (Yahoo Finance) — per account")
        st.caption("If enabled for an account, the fixed annual rate is ignored for that account.")

        market_cfgs = {}
        default_start = date(2010, 1, 1)

        for acct in ["IRA", "401K", "529", "ETF"]:
            with st.expander(f"{acct} Market Mode", expanded=(acct == "ETF")):
                enabled = st.checkbox(f"Enable Market Mode for {acct}", value=(acct == "ETF"), key=f"mm_{acct}_en")
                ticker = st.text_input("Ticker (Yahoo symbol)", value="SPY" if acct in ("IRA", "401K", "ETF") else "VOO",
                                       key=f"mm_{acct}_t", disabled=(not enabled))
                start_dt = st.date_input("Backtest start date", value=default_start, key=f"mm_{acct}_sd", disabled=(not enabled))
                buy_days = st.multiselect(
                    "Buy on (trading day of month)",
                    options=["first", "15th"],
                    default=["first", "15th"],
                    key=f"mm_{acct}_bd",
                    disabled=(not enabled),
                )
                invest_initial = st.checkbox("Invest initial balance into the ticker", value=True, key=f"mm_{acct}_ii", disabled=(not enabled))

                market_cfgs[acct] = {
                    "enabled": bool(enabled),
                    "ticker": (ticker or "").strip().upper(),
                    "start_date": start_dt,
                    "buy_on": tuple(buy_days) if buy_days else ("first", "15th"),
                    "invest_initial": bool(invest_initial),
                }

    # -----------------------
    # MAIN: Targets / Run
    # -----------------------
    st.subheader("Targets (\"hunt\" goals)")
    tcol1, tcol2, tcol3, tcol4, tcol5 = st.columns(5)
    with tcol1:
        t_ira = st.number_input("Target: IRA", min_value=0.0, value=st.session_state.get("target_ira", 1_000_000.0),
                                step=50_000.0, key="target_ira")
    with tcol2:
        t_401k = st.number_input("Target: 401K", min_value=0.0, value=st.session_state.get("target_401k", 2_000_000.0),
                                 step=50_000.0, key="target_401k")
    with tcol3:
        t_529 = st.number_input("Target: 529", min_value=0.0, value=st.session_state.get("target_529", 250_000.0),
                                step=10_000.0, key="target_529")
    with tcol4:
        t_etf = st.number_input("Target: ETF", min_value=0.0, value=st.session_state.get("target_etf", 500_000.0),
                                step=25_000.0, key="target_etf")
    with tcol5:
        use_total_target = st.checkbox("Use Total target", value=st.session_state.get("use_total_target", False), key="use_total_target")
        t_total = st.number_input("Target: Total", min_value=0.0, value=st.session_state.get("target_total", 4_000_000.0),
                                  step=100_000.0, disabled=(not use_total_target), key="target_total")

    targets = {"IRA": float(t_ira), "401K": float(t_401k), "529": float(t_529), "ETF": float(t_etf)}
    if use_total_target:
        targets["Total"] = float(t_total)

    st.subheader("What counts for 'Retirement'?")
    req_default = [k for k in ["IRA", "401K", "ETF"] if k in targets]
    require = st.multiselect("Required targets to declare retirement", options=list(targets.keys()), default=req_default, key="require_targets")

    run = st.button("Run Hunt", type="primary")

    # -----------------------
    # RUN + PERSIST RESULTS
    # -----------------------
    if run:
        start_balances = {"IRA": float(ira_s), "401K": float(k401_s), "529": float(c529_s), "ETF": float(etf_s)}
        annual_rates = {"IRA": float(ira_i), "401K": float(k401_i), "529": float(c529_i), "ETF": float(etf_i)}

        overrides = {}
        dcas = {}

        # Baseline run to get monthly contributions schedule (independent of returns)
        tmp = simulate_accounts(
            start_balances=start_balances,
            annual_rates_pct=annual_rates,
            N_years=int(horizon_years),
            phases=phases,
            periods_per_year=periods_per_year,
            overrides=None,
        )
        contribs = tmp["contribs"]  # (n_months, 4)

        acct_to_col = {"IRA": 0, "401K": 1, "529": 2, "ETF": 3}

        with st.spinner("Running market backtests (Yahoo Finance) for enabled accounts..."):
            for acct in ["IRA", "401K", "529", "ETF"]:
                cfg = market_cfgs.get(acct, {})
                if not cfg.get("enabled", False):
                    continue

                ticker = cfg.get("ticker", "").strip()
                if not ticker:
                    st.warning(f"{acct} Market Mode is enabled but ticker is blank — skipping.")
                    continue

                monthly = contribs[:, acct_to_col[acct]].astype(float)
                invest_initial = bool(cfg.get("invest_initial", True))
                initial_amt = float(start_balances[acct]) if invest_initial else 0.0

                if (not invest_initial) and start_balances[acct] > 0:
                    st.warning(
                        f"{acct}: 'Invest initial balance' is OFF. "
                        "That starting balance will be excluded from the market backtest (only monthly contributions invested)."
                    )

                try:
                    dca = cached_dca(
                        ticker=ticker,
                        start_date=cfg.get("start_date", date(2010, 1, 1)),
                        N_years=int(horizon_years),
                        monthly_contribs=tuple(float(x) for x in monthly.tolist()),
                        buy_on=tuple(cfg.get("buy_on", ("first", "15th"))),
                        invest_initial_balance=initial_amt,
                    )
                    overrides[acct] = dca["monthly_values"]
                    dcas[acct] = dca

                    st.info(
                        f"{acct} market mode: {ticker} (data {dca['start_used']} → {dca['end_used']}); "
                        f"buys on {list(cfg.get('buy_on'))}"
                    )

                except Exception as e:
                    st.error(f"{acct} market mode failed ({ticker}): {e}. Falling back to fixed rate for {acct}.")

        sim = simulate_accounts(
            start_balances=start_balances,
            annual_rates_pct=annual_rates,
            N_years=int(horizon_years),
            phases=phases,
            periods_per_year=periods_per_year,
            overrides=overrides if overrides else None,
        )

        yrs = crossing_years(sim, targets)
        retire_year = retirement_year_from_targets(sim, targets, require=require if require else None)

        st.session_state["last_run_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state["sim"] = sim
        st.session_state["yrs"] = yrs
        st.session_state["retire_year"] = retire_year
        st.session_state["targets"] = targets
        st.session_state["require"] = require
        st.session_state["phases"] = phases
        st.session_state["start_balances_dict"] = start_balances
        st.session_state["annual_rates_dict"] = annual_rates
        st.session_state["horizon_years_int"] = int(horizon_years)
        st.session_state["market_cfgs"] = market_cfgs
        st.session_state["dcas"] = dcas

    # -----------------------
    # DISPLAY LAST RUN (if any)
    # -----------------------
    if "sim" not in st.session_state:
        st.info("Set your inputs and click **Run Hunt** to generate results.")
    else:
        st.caption(f"Showing results from last run: {st.session_state.get('last_run_at','')}")

        sim = st.session_state["sim"]
        yrs = st.session_state["yrs"]
        retire_year = st.session_state["retire_year"]
        targets = st.session_state["targets"]
        require = st.session_state["require"]
        phases_used = st.session_state.get("phases", [])
        sb = st.session_state.get("start_balances_dict", {})
        ar = st.session_state.get("annual_rates_dict", {})
        N_years = int(st.session_state.get("horizon_years_int", sim.get("N_years", 1)))
        market_cfgs_used = st.session_state.get("market_cfgs", {})
        dcas = st.session_state.get("dcas", {})

        left, right = st.columns([1, 1], gap="large")

        with left:
            st.subheader("Crossing Years")
            df = pd.DataFrame([{"Account": k, "Target": float(targets[k]), "Crossing Year": yrs.get(k)} for k in targets.keys()])
            df["Target"] = df["Target"].map(lambda x: f"${float(x):,.0f}")
            df["Crossing Year"] = df["Crossing Year"].map(lambda x: "" if x is None else f"{float(x):.2f}")
            st.dataframe(df, use_container_width=True)

            st.subheader("Retirement Year")
            if retire_year is None:
                st.error("Not all required targets were reached within the simulated horizon.")
            else:
                st.success(f"All required targets met by **Year {retire_year:.2f}**")
                st.caption(f"(Retirement year = max crossing year across: {require if require else list(targets.keys())})")

            st.subheader("Final Report")
            show_raw = st.checkbox("Show raw numbers (JSON)", value=False, key="show_raw_final")
            if show_raw:
                st.json(sim["final_report"])
            else:
                fr = sim["final_report"]
                fb = fr["Final Balances"]
                tc = fr["Total Contributions"]
                gr = fr["Growth"]

                summary_df = pd.DataFrame(
                    [
                        {"Metric": "Final Balances", **{k: money(v) for k, v in fb.items()}},
                        {"Metric": "Total Contributions", **{k: money(v) for k, v in tc.items()}},
                        {"Metric": "Growth", **{k: money(v) for k, v in gr.items()}},
                    ]
                )
                cols = ["Metric", "IRA", "401K", "529", "ETF", "Total"]
                summary_df = summary_df[[c for c in cols if c in summary_df.columns]]
                st.dataframe(summary_df, use_container_width=True)

            # Goal Solver (windowed)
            with st.expander("Goal Solver: extra monthly needed to hit a target by a year", expanded=False):
                acct = st.selectbox("Target account", options=["IRA", "401K", "529", "ETF", "Total"], index=0, key="gs_acct")
                goal_year = st.number_input("Goal year", min_value=1, max_value=int(N_years), value=int(min(N_years, 20)), key="gs_year")
                goal_value = st.number_input(
                    "Goal value ($)", min_value=0.0,
                    value=float(targets.get(acct, 0.0) if acct != "Total" else 1_000_000.0),
                    step=10_000.0, key="gs_value"
                )

                extra_to_default = acct if acct != "Total" else "ETF"
                extra_to = st.selectbox(
                    "Apply extra monthly contributions to",
                    options=["IRA", "401K", "529", "ETF"],
                    index=["IRA", "401K", "529", "ETF"].index(extra_to_default),
                    key="gs_extra_to",
                )

                extra_start_year = st.number_input(
                    "Start extra contributions at year",
                    min_value=0,
                    max_value=int(goal_year - 1) if goal_year > 1 else 0,
                    value=st.session_state.get("gs_start", 0),
                    step=1,
                    key="gs_start",
                )
                extra_end_year = st.number_input(
                    "Stop extra contributions at year (exclusive)",
                    min_value=int(extra_start_year + 1),
                    max_value=int(goal_year),
                    value=int(goal_year),
                    step=1,
                    key="gs_end",
                )

                any_market = any(bool(market_cfgs_used.get(a, {}).get("enabled", False)) for a in ["IRA", "401K", "529", "ETF"])
                if any_market:
                    st.warning(
                        "Goal Solver uses the fixed-rate model (not the Yahoo market backtests). "
                        "Turn off Market Mode if you want the solver to match the backtests exactly."
                    )

                if st.button("Solve extra monthly required", type="secondary", key="gs_solve"):
                    try:
                        res = solve_extra_monthly_contribution(
                            start_balances=sb,
                            annual_rates_pct=ar,
                            phases=phases_used,
                            N_years=N_years,
                            target_key=acct,
                            target_value=float(goal_value),
                            goal_year=int(goal_year),
                            extra_to=str(extra_to),
                            extra_start_year=int(extra_start_year),
                            extra_end_year=int(extra_end_year),
                            tol=5.0,
                            max_iter=35,
                        )

                        st.write(f"Baseline at Year {res['goal_year']}: **{money(res['baseline_value'])}**")
                        st.write(f"Target: **{money(res['target_value'])}**")
                        st.write(f"Shortfall: **{money(res['shortfall'])}**")

                        if res.get("extra_monthly_required") is None:
                            st.error("Could not find an extra monthly contribution that reaches the target within the search bounds.")
                            if "note" in res:
                                st.caption(res["note"])
                        else:
                            st.success(
                                f"Extra needed: **{money(res['extra_monthly_required'])} / month** "
                                f"(applied to {res['extra_to']} from years {res['extra_start_year']}–{res['extra_end_year']})"
                            )
                            st.caption(f"Check: achieved at goal year ≈ {money(res['achieved_value'])} (iterations: {res['iterations']})")

                    except Exception as e:
                        st.error(f"Solver error: {e}")

            if hasattr(sim["phase_reports"], "empty") and (not sim["phase_reports"].empty):
                st.subheader("Balances at End of Each Phase")
                st.dataframe(sim["phase_reports"], use_container_width=True)

        with right:
            st.subheader("Growth Chart")
            embed_summary = st.checkbox("Embed summary text in chart (exports can look busy)", value=False)
            fig = make_growth_figure(sim, embed_summary=embed_summary)

            if retire_year is not None:
                ax = fig.axes[0]
                ax.axvline(x=retire_year, linestyle="--", linewidth=2, alpha=0.8)
                y_total = float(np.interp(retire_year, sim["years_x"], sim["total"]))
                ax.annotate(
                    f"RETIRE\nYear {retire_year:.2f}",
                    xy=(retire_year, y_total),
                    xytext=(retire_year + 0.5, y_total * 0.9),
                    arrowprops=dict(arrowstyle="->", lw=1),
                    fontsize=9,
                    ha="left",
                )

            try:
                st.pyplot(fig, clear_figure=False, use_container_width=True)
            except TypeError:
                st.pyplot(fig, clear_figure=False)

            # Market-mode detail expanders per account
            for acct in ["IRA", "401K", "529", "ETF"]:
                if acct in dcas:
                    dca = dcas[acct]
                    cfg = market_cfgs_used.get(acct, {})
                    with st.expander(f"{acct} Market Mode details (prices + trades)", expanded=False):
                        st.write(f"Ticker: **{dca.get('ticker','')}**")
                        st.write(f"Data used: {dca.get('start_used','')} → {dca.get('end_used','')}")
                        st.write(f"Buy on: **{list(cfg.get('buy_on', []))}** | Invest initial: **{cfg.get('invest_initial', True)}**")

                        trades_df = dca.get("trades")
                        if trades_df is not None and hasattr(trades_df, "empty") and (not trades_df.empty):
                            st.subheader("Trades")
                            st.dataframe(trades_df, use_container_width=True)
                            csv_bytes = trades_df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                f"Download {acct} trades CSV",
                                data=csv_bytes,
                                file_name=f"{dca.get('ticker','TICKER')}_{acct}_trades.csv",
                                mime="text/csv",
                            )
                        else:
                            st.info("No trades were generated (likely $0 contributions for this account).")

            st.subheader("Downloads")
            c_png, c_chart_pdf, c_report_pdf = st.columns(3)

            with c_png:
                buf_png = io.BytesIO()
                fig.savefig(buf_png, format="png", dpi=300, bbox_inches="tight", facecolor="white")
                buf_png.seek(0)
                st.download_button("Download chart as PNG", data=buf_png, file_name="investment_growth.png", mime="image/png")

            with c_chart_pdf:
                buf_pdf = io.BytesIO()
                fig.savefig(buf_pdf, format="pdf", bbox_inches="tight", facecolor="white")
                buf_pdf.seek(0)
                st.download_button("Download chart as PDF", data=buf_pdf, file_name="investment_growth.pdf", mime="application/pdf")

            with c_report_pdf:
                # Build market mode payload for PDF
                market_mode_payload = {}
                for acct, dca in dcas.items():
                    cfg = market_cfgs_used.get(acct, {})
                    market_mode_payload[acct] = {
                        "enabled": True,
                        "ticker": dca.get("ticker", ""),
                        "start_used": dca.get("start_used", ""),
                        "end_used": dca.get("end_used", ""),
                        "buy_on": list(cfg.get("buy_on", [])),
                        "invest_initial": bool(cfg.get("invest_initial", True)),
                        "trades": dca.get("trades"),
                    }

                report_bytes = build_report_pdf_bytes(
                    sim=sim,
                    targets=targets,
                    crossing=yrs,
                    retire_year=retire_year,
                    require=require if require else None,
                    fig=fig,
                    market_mode=market_mode_payload if market_mode_payload else None,
                )
                st.download_button(
                    "Download full report PDF",
                    data=report_bytes,
                    file_name="retirement_hunt_report.pdf",
                    mime="application/pdf",
                )

# ======================================================================================
# TAB 2: FY26 Baseline (Excel)
# ======================================================================================
with tabs[1]:
    st.header("FY26 Army Income / Budget / Investment Baseline (Excel)")
    st.caption("This tab reads your spreadsheet, lets you scale selected rows, and can push values into the Retirement Hunt inputs.")

    default_path = str(DEFAULT_BASELINE_XLSX) if DEFAULT_BASELINE_XLSX.exists() else ""
    uploaded = st.file_uploader("Upload the FY26 baseline spreadsheet (optional)", type=["xlsx"], key="upl_baseline")

    if uploaded is not None:
        path = uploaded
        sheets = pd.ExcelFile(uploaded).sheet_names
        dfs = {name: pd.read_excel(uploaded, sheet_name=name, header=None) for name in sheets}
    elif default_path:
        dfs = load_excel(default_path)
        sheets = list(dfs.keys())
    else:
        st.warning("Baseline spreadsheet not found. Upload it above.")
        dfs = {}
        sheets = []

    if dfs:
        sheet = st.selectbox("Sheet", options=sheets, index=(0 if "Stacking Bread" not in sheets else sheets.index("Stacking Bread")))
        df_raw = dfs[sheet]
        if sheet == "Stacking Bread":
            sb_df = parse_stacking_bread(df_raw)

            if sb_df.empty:
                st.error("Could not parse the 'Stacking Bread' sheet.")
            else:
                st.subheader("Parsed table")
                years = [int(c) for c in sb_df.columns]
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    y0 = st.selectbox("Choose a year column to extract contributions", options=years, index=0)
                with c2:
                    scale_rows = st.multiselect("Rows to scale (+%)", options=list(sb_df.index), default=[])
                with c3:
                    pct = st.number_input("Scale percent", value=0.0, step=1.0)
                with c4:
                    scale_all_years = st.checkbox("Scale all year columns", value=True)

                sb_work = sb_df.copy()
                if scale_rows and pct != 0:
                    cols_to_scale = list(sb_work.columns) if scale_all_years else [str(y0)]
                    sb_work.loc[scale_rows, cols_to_scale] = sb_work.loc[scale_rows, cols_to_scale] * (1.0 + float(pct) / 100.0)

                # Display a focused view: select a section by keyword filter
                filter_text = st.text_input("Filter rows (contains)", value="")
                view = sb_work
                if filter_text.strip():
                    mask = view.index.to_series().str.contains(filter_text.strip(), case=False, na=False)
                    view = view.loc[mask]

                years_to_show = st.multiselect("Year columns to show", options=years, default=[y0])
                cols_show = [str(int(y)) for y in years_to_show] if years_to_show else [str(y0)]

                st.markdown("#### Edit baseline values (affects extraction + calculations)")
                st.caption("Click into a cell to edit. Edits are saved automatically and used below.")

                sb_key = f"baseline_sb_{sheet}"
                sb_orig_key = f"baseline_sb_original_{sheet}"

                # Store originals once
                if sb_orig_key not in st.session_state:
                    st.session_state[sb_orig_key] = sb_df.copy()
                if sb_key not in st.session_state:
                    st.session_state[sb_key] = sb_df.copy()

                # Work from persisted baseline
                sb_persist = st.session_state[sb_key].copy()
                sb_persist = sb_persist.reindex(index=sb_df.index, columns=sb_df.columns)

                # Actions
                a1, a2 = st.columns([1, 1])
                with a1:
                    if st.button("Apply % increase to selected rows", type="secondary", key=f"baseline_apply_scale_{sheet}"):
                        if scale_rows and pct != 0:
                            cols_to_scale = list(sb_persist.columns) if scale_all_years else [str(y0)]
                            sb_persist.loc[scale_rows, cols_to_scale] = sb_persist.loc[scale_rows, cols_to_scale] * (1.0 + float(pct) / 100.0)
                            st.session_state[sb_key] = sb_persist
                            st.success("Applied percent increase.")
                            st.rerun()
                        else:
                            st.info("Select at least one row and set a non-zero percent.")
                with a2:
                    if st.button("Reset baseline edits from Excel", type="secondary", key=f"baseline_reset_{sheet}"):
                        st.session_state[sb_key] = st.session_state[sb_orig_key].copy()
                        st.success("Reset to original values.")
                        st.rerun()

                # Apply filter for display/editing
                view2 = sb_persist
                if filter_text.strip():
                    mask = view2.index.to_series().str.contains(filter_text.strip(), case=False, na=False)
                    view2 = view2.loc[mask]

                display_df = view2[cols_show].copy().reset_index().rename(columns={"index": "Metric"})
                display_df.insert(0, "Metric", display_df.pop("Metric"))

                edited_df = st.data_editor(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={c: st.column_config.NumberColumn(format="%.2f") for c in cols_show},
                    disabled=["Metric"],
                    key=f"baseline_editor_{sheet}",
                )

                # Persist edits automatically (only visible subset)
                tmp = edited_df.set_index("Metric")
                for c in cols_show:
                    tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

                for metric in tmp.index:
                    for c in cols_show:
                        v = tmp.loc[metric, c]
                        if not pd.isna(v):
                            sb_persist.loc[metric, c] = float(v)

                st.session_state[sb_key] = sb_persist

                # Use persisted (edited + scaled) for extraction below
                sb_work = sb_persist

                st.divider()
                st.subheader("Extract contributions for Retirement Hunt (from this sheet/year)")

                # Defaults
                # Find likely rows
                def first_match(substr):
                    for r in sb_work.index:
                        if substr.lower() in str(r).lower():
                            return r
                    return None

                row_401k = first_match("TSP Contribution")
                row_ira = first_match("Your ROTH IRA Contribution")
                row_etf = first_match("After TSP Remaining")

                colA, colB = st.columns(2)
                with colA:
                    pick_401k = st.selectbox("Annual 401K contribution row", options=list(sb_work.index), index=(list(sb_work.index).index(row_401k) if row_401k in sb_work.index else 0))
                    pick_ira = st.selectbox("Annual IRA contribution row", options=list(sb_work.index), index=(list(sb_work.index).index(row_ira) if row_ira in sb_work.index else 0))
                    pick_etf = st.selectbox("Annual ETF/Brokerage contribution row (optional)", options=["(none)"] + list(sb_work.index),
                                            index=(1 + list(sb_work.index).index(row_etf) if row_etf in sb_work.index else 0))
                with colB:
                    mult = st.number_input("Multiply extracted contributions by", min_value=0.0, value=1.0, step=0.05)
                    force_phases_1 = st.checkbox("Force Retirement Hunt to 1 phase (overwrite Phase 1 inputs)", value=True)

                def val(rowname):
                    if rowname is None or rowname == "(none)":
                        return 0.0
                    v = sb_work.loc[rowname, str(y0)]
                    return 0.0 if pd.isna(v) else float(v)

                annual_401k = val(pick_401k) * float(mult)
                annual_ira = val(pick_ira) * float(mult)
                annual_etf = val(pick_etf) * float(mult) if pick_etf != "(none)" else 0.0

                st.write(f"Annual 401K: **{money(annual_401k)}**  → Monthly: **{money(annual_401k/12)}**")
                st.write(f"Annual IRA: **{money(annual_ira)}**  → Monthly: **{money(annual_ira/12)}**")
                st.write(f"Annual ETF: **{money(annual_etf)}**  → Monthly: **{money(annual_etf/12)}**")

                if st.button("Send these values to Retirement Hunt Phase 1", type="secondary"):
                    updates = {}
                    if force_phases_1:
                        updates["num_phases"] = 1
                        updates["p0_sy"] = 0
                        updates["p0_ey"] = int(st.session_state.get("horizon_years", 40))

                    updates["p0_tsp"] = float(annual_401k / 12.0)
                    updates["p0_ira"] = float(annual_ira / 12.0)
                    updates["p0_etf"] = float(annual_etf / 12.0)

                    queue_state_updates(updates)

        else:
            st.subheader("Raw sheet preview")
            st.dataframe(df_raw.head(50), use_container_width=True)

# ======================================================================================
# TAB 3: Conscious Spending Plan (Excel)
# ======================================================================================
with tabs[2]:
    st.header("IWT Conscious Spending Plan (Excel)")
    st.caption("Edit the plan, apply percentage increases, compute monthly surplus, and push investment allocations into Phase 1.")

    default_path = str(DEFAULT_CSP_XLSX) if DEFAULT_CSP_XLSX.exists() else ""
    uploaded = st.file_uploader("Upload the Conscious Spending Plan spreadsheet (optional)", type=["xlsx"], key="upl_csp")

    if uploaded is not None:
        sheets = pd.ExcelFile(uploaded).sheet_names
        dfs = {name: pd.read_excel(uploaded, sheet_name=name, header=None) for name in sheets}
    elif default_path:
        dfs = load_excel(default_path)
        sheets = list(dfs.keys())
    else:
        st.warning("Conscious Spending Plan spreadsheet not found. Upload it above.")
        dfs = {}
        sheets = []

    if dfs:
        default_sheet = "Conscious Spending Plan" if "Conscious Spending Plan" in sheets else sheets[0]
        sheet = st.selectbox("Sheet", options=sheets, index=sheets.index(default_sheet))
        df_raw = dfs[sheet]
        plan = parse_conscious_plan(df_raw)

        if plan.empty:
            st.warning("No editable line-items found in this sheet. (Try 'Conscious Spending Plan' or an Example sheet.)")
            st.dataframe(df_raw.head(80), use_container_width=True)
        else:
            # Store editable in session_state
            key = f"csp_plan_{sheet}"
            if key not in st.session_state:
                st.session_state[key] = plan

            plan_edit = st.session_state[key].copy()

            st.subheader("Editable plan (amounts are monthly)")
            # bulk scaling controls
            c1, c2, c3 = st.columns(3)
            with c1:
                section = st.selectbox("Section to increase", options=["All"] + sorted([s for s in plan_edit["Section"].unique() if s != ""]), index=0)
            with c2:
                pct = st.number_input("Increase by (%)", value=0.0, step=1.0)
            with c3:
                include_zero = st.checkbox("Also scale $0 rows", value=False)

            if st.button("Apply increase", type="secondary"):
                plan_edit = apply_percent_to_items(plan_edit, section, pct, include_zero=include_zero)
                st.session_state[key] = plan_edit
                st.success("Applied.")

            # editor
            edited = st.data_editor(
                plan_edit,
                use_container_width=True,
                num_rows="dynamic",
                column_config={
                    "Amount": st.column_config.NumberColumn(format="%.2f", step=10.0),
                },
                key=f"editor_{sheet}",
            )
            # Persist edits
            st.session_state[key] = edited

            # Compute totals
            dfp = edited.copy()
            dfp["Amount"] = pd.to_numeric(dfp["Amount"], errors="coerce").fillna(0.0)

            # Pull key items
            def get_amount_by_item(substr):
                m = dfp["Item"].str.contains(substr, case=False, na=False)
                if not m.any():
                    return 0.0
                return float(dfp.loc[m, "Amount"].iloc[0])

            net_income = get_amount_by_item("Net monthly income")
            gross_income = get_amount_by_item("Gross monthly income")

            fixed_total = float(dfp.loc[dfp["Section"].str.contains("FIXED COSTS", case=False, na=False), "Amount"].sum())
            invest_total = float(dfp.loc[dfp["Section"].str.contains("INVESTMENTS", case=False, na=False), "Amount"].sum())
            savings_total = float(dfp.loc[dfp["Section"].str.contains("SAVINGS GOALS", case=False, na=False), "Amount"].sum())
            guilt_total = float(dfp.loc[dfp["Section"].str.contains("GUILT-FREE", case=False, na=False), "Amount"].sum())

            if net_income == 0.0:
                # If user hasn't filled that row, infer from gross if present
                net_income = gross_income

            surplus = net_income - (fixed_total + invest_total + savings_total + guilt_total)

            st.subheader("Monthly summary")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Net income", money(net_income))
            m2.metric("Fixed costs", money(fixed_total))
            m3.metric("Investments", money(invest_total))
            m4.metric("Savings goals", money(savings_total))
            m5.metric("Guilt-free", money(guilt_total))

            st.metric("Unallocated surplus", money(surplus))

            st.divider()
            st.subheader("Push Investments into Retirement Hunt Phase 1")
            st.caption("Allocate the 'Investments' monthly total across IRA / 401K / 529 / ETF.")

            alloc_total = invest_total
            if alloc_total <= 0:
                st.warning("Investments total is $0 — fill in investment line-items above, then allocate.")
            else:
                a1, a2, a3, a4 = st.columns(4)
                with a1:
                    w_ira = st.slider("IRA share", 0.0, 1.0, 0.25, 0.05)
                with a2:
                    w_401k = st.slider("401K share", 0.0, 1.0, 0.50, 0.05)
                with a3:
                    w_529 = st.slider("529 share", 0.0, 1.0, 0.10, 0.05)
                with a4:
                    w_etf = st.slider("ETF share", 0.0, 1.0, 0.15, 0.05)

                w_sum = w_ira + w_401k + w_529 + w_etf
                if abs(w_sum - 1.0) > 1e-6:
                    st.error(f"Shares must sum to 1.0 (currently {w_sum:.2f}).")
                else:
                    ira_m = alloc_total * w_ira
                    k401_m = alloc_total * w_401k
                    c529_m = alloc_total * w_529
                    etf_m = alloc_total * w_etf

                    st.write(
                        f"IRA **{money(ira_m)}** | 401K **{money(k401_m)}** | 529 **{money(c529_m)}** | ETF **{money(etf_m)}**"
                    )

                    overwrite = st.checkbox("Overwrite Phase 1 contributions", value=True)
                    if st.button("Send allocation to Retirement Hunt Phase 1", type="secondary"):
                        if overwrite:
                            updates = {
                                "p0_ira": float(ira_m),
                                "p0_tsp": float(k401_m),
                                "p0_529": float(c529_m),
                                "p0_etf": float(etf_m),
                            }
                            queue_state_updates(updates)
                        else:
                            st.info("Overwrite disabled — no changes made.")
