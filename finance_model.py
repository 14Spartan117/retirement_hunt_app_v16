from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

# Canonical account order used everywhere (matches history columns)
ACCOUNTS = ["IRA", "401K", "529", "ETF"]


@dataclass(frozen=True)
class Phase:
    """
    Contribution phase:
      - start_year inclusive
      - end_year exclusive (like Python range)
      - monthly contributions per account
    """
    phase: int
    start_year: int
    end_year: int
    ira: float
    tsp: float
    college: float
    etf: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "start_year": self.start_year,
            "end_year": self.end_year,
            "monthly_contributions": {
                "IRA": self.ira,
                "401K": self.tsp,
                "529": self.college,
                "ETF": self.etf,
            },
        }


def _build_contribs(
    N_years: int,
    periods_per_year: int,
    phases: List[Phase],
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Build month-by-month contribution matrix and phase bounds metadata.
    contribs[t, j] = contribution for month t to account j (ACCOUNTS order)
    """
    n = N_years * periods_per_year
    contribs = np.zeros((n, 4), dtype=float)

    phase_bounds: List[Dict[str, Any]] = []
    for p in phases:
        start_idx = max(0, p.start_year * periods_per_year)
        end_idx = min(n, p.end_year * periods_per_year)

        if start_idx < end_idx:
            contribs[start_idx:end_idx, :] = [p.ira, p.tsp, p.college, p.etf]
            phase_bounds.append(
                {
                    "phase": p.phase,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "start_year": p.start_year,
                    "end_year": p.end_year,
                    "end_month_idx": end_idx - 1,
                }
            )

    phase_bounds.sort(key=lambda d: d["phase"])
    return contribs, phase_bounds


def simulate_accounts(
    start_balances: Dict[str, float],
    annual_rates_pct: Dict[str, float],
    N_years: int,
    phases: List[Phase],
    periods_per_year: int = 12,
    overrides: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Simulate balances with monthly compounding + phased monthly contributions.

    overrides:
      - Pass overrides={"ETF": monthly_values_array} to replace ETF values after the base simulation.
      - Override arrays must have length n_months = N_years * periods_per_year.
    """
    if N_years <= 0:
        raise ValueError("N_years must be positive.")

    initial_balances = np.array([float(start_balances[k]) for k in ACCOUNTS], dtype=float)
    rates = np.array([float(annual_rates_pct[k]) for k in ACCOUNTS], dtype=float)

    n = N_years * periods_per_year
    contribs, phase_bounds = _build_contribs(N_years, periods_per_year, phases)

    # Monthly compounding factors
    monthly = 1.0 + (rates / 1200.0)
    s1 = np.diag(monthly)

    s2 = initial_balances.reshape(4, 1)
    history = np.zeros((n, 4), dtype=float)

    for t in range(n):
        s3 = s1 @ s2
        s5 = s3 + contribs[t].reshape(4, 1)
        history[t, :] = s5.flatten()
        s2 = s5

    # Apply overrides (replace columns AFTER the base simulation)
    if overrides:
        for acct, series in overrides.items():
            if acct not in ACCOUNTS:
                raise ValueError(f"Unknown override account: {acct}. Must be one of {ACCOUNTS}.")
            series = np.asarray(series, dtype=float).reshape(-1)
            if len(series) != n:
                raise ValueError(f"Override for {acct} has length {len(series)} but expected {n}.")
            j = ACCOUNTS.index(acct)
            history[:, j] = series

    total = history.sum(axis=1)

    # x-axis in years
    months = np.arange(1, n + 1)
    years_x = months / periods_per_year

    # phase reports (end of each phase)
    phase_rows: List[Dict[str, Any]] = []
    for b in phase_bounds:
        m = int(b["end_month_idx"])
        phase_year = (m + 1) / periods_per_year
        bal = history[m, :]
        phase_rows.append(
            {
                "Phase": int(b["phase"]),
                "Year": float(phase_year),
                "IRA": float(bal[0]),
                "401K": float(bal[1]),
                "529": float(bal[2]),
                "ETF": float(bal[3]),
                "Total": float(bal.sum()),
            }
        )
    phase_reports = pd.DataFrame(phase_rows)

    # summary numbers
    final = history[-1, :]
    total_contribs_per_account = contribs.sum(axis=0)
    total_contribs = float(total_contribs_per_account.sum())
    growth = final - (initial_balances + total_contribs_per_account)

    final_report = {
        "Years Simulated": int(N_years),
        "Final Balances": {
            "IRA": float(final[0]),
            "401K": float(final[1]),
            "529": float(final[2]),
            "ETF": float(final[3]),
            "Total": float(final.sum()),
        },
        "Total Contributions": {
            "IRA": float(total_contribs_per_account[0]),
            "401K": float(total_contribs_per_account[1]),
            "529": float(total_contribs_per_account[2]),
            "ETF": float(total_contribs_per_account[3]),
            "Total": total_contribs,
        },
        "Growth": {
            "IRA": float(growth[0]),
            "401K": float(growth[1]),
            "529": float(growth[2]),
            "ETF": float(growth[3]),
            "Total": float(growth.sum()),
        },
    }

    input_summary = {
        "Starting Balances": dict(start_balances),
        "Interest Rates (%)": dict(annual_rates_pct),
    }

    return {
        "history": history,
        "total": total,
        "years_x": years_x,
        "contribs": contribs,
        "phase_bounds": phase_bounds,
        "phase_reports": phase_reports,
        "final_report": final_report,
        "input_summary": input_summary,
        "phases": [p.to_dict() for p in phases],
        "N_years": int(N_years),
        "periods_per_year": int(periods_per_year),
        "annual_rates_pct": dict(annual_rates_pct),
        "start_balances": dict(start_balances),
    }


def make_growth_figure(sim: Dict[str, Any], embed_summary: bool = False) -> plt.Figure:
    """
    Build and return matplotlib Figure (Streamlit: st.pyplot(fig)).
    """
    history = sim["history"]
    total = sim["total"]
    years_x = sim["years_x"]
    N = sim["N_years"]
    phase_reports = sim["phase_reports"]
    rates = sim["annual_rates_pct"]
    sb = sim["start_balances"]
    phases = sim["phases"]

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(years_x, history[:, 0], label=f"IRA ({rates['IRA']:.1f}%/yr)", linewidth=2)
    ax.plot(years_x, history[:, 1], label=f"401K ({rates['401K']:.1f}%/yr)", linewidth=2)
    ax.plot(years_x, history[:, 2], label=f"529 ({rates['529']:.1f}%/yr)", linewidth=2)
    ax.plot(years_x, history[:, 3], label=f"ETF ({rates['ETF']:.1f}%/yr)", linewidth=2)
    ax.plot(years_x, total, label="Total Value", linestyle="--", linewidth=2)

    ax.set_title(f"Growth of IRA, 401K, ETF, 529 Over {N} Years (Monthly Compounding)", fontsize=13)
    ax.set_xlabel("Years", fontsize=11)
    ax.set_ylabel("Value", fontsize=11)
    ax.legend()
    ax.grid(True)
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

    # End-of-phase annotations
    if hasattr(phase_reports, "empty") and not phase_reports.empty:
        for _, r in phase_reports.iterrows():
            phase_year = float(r["Year"])
            total_val = float(r["Total"])
            ax.axvline(x=phase_year, linestyle=":", linewidth=1.5, alpha=0.7)
            ax.annotate(
                f"End Phase {int(r['Phase'])}\n\\${total_val:,.0f}",
                xy=(phase_year, total_val),
                xytext=(phase_year + 0.3, total_val * 0.95),
                arrowprops=dict(arrowstyle="->", lw=1),
                fontsize=8,
                ha="left",
            )

    fig.tight_layout(rect=(0, 0.34, 1, 1))

    if embed_summary:
        final = sim["final_report"]["Final Balances"]
        summary_str = "Initial Inputs:\n"
        summary_str += f"  IRA:   \\${sb['IRA']:,.0f}\n"
        summary_str += f"  401K:  \\${sb['401K']:,.0f}\n"
        summary_str += f"  529:   \\${sb['529']:,.0f}\n"
        summary_str += f"  ETF:   \\${sb['ETF']:,.0f}\n\n"

        summary_str += "Contribution Phases:\n"
        for p in phases:
            c = p["monthly_contributions"]
            summary_str += (
                f"  Phase {p['phase']} (Years {p['start_year']}-{p['end_year']}): "
                f"IRA \\${c['IRA']:,.0f}, 401K \${c['401K']:,.0f}, 529 \${c['529']:,.0f}, ETF \${c['ETF']:,.0f}\n"
            )

        summary_str += "\nFinal Balances:\n"
        summary_str += f"  IRA:   \\${final['IRA']:,.0f}\n"
        summary_str += f"  401K:  \\${final['401K']:,.0f}\n"
        summary_str += f"  529:   \\${final['529']:,.0f}\n"
        summary_str += f"  ETF:   \\${final['ETF']:,.0f}\n"
        summary_str += f"  TOTAL: \\${final['Total']:,.0f}\n"

        fig.text(0.01, 0.01, summary_str, ha="left", va="bottom", fontsize=8, family="monospace")

    return fig


# ------------------------------
# Hunt helpers
# ------------------------------

def _first_crossing_year(values_1d: np.ndarray, target: float, periods_per_year: int) -> Optional[float]:
    idx = np.where(values_1d >= float(target))[0]
    if idx.size == 0:
        return None
    month_idx = int(idx[0])
    return (month_idx + 1) / periods_per_year


def crossing_years(sim: Dict[str, Any], targets: Dict[str, float]) -> Dict[str, Optional[float]]:
    ppy = int(sim["periods_per_year"])
    hist = sim["history"]
    total = sim["total"]

    out: Dict[str, Optional[float]] = {}
    for k, tgt in targets.items():
        if tgt is None:
            continue
        tgt_f = float(tgt)
        if k == "Total":
            out["Total"] = _first_crossing_year(total, tgt_f, ppy)
        elif k in ACCOUNTS:
            j = ACCOUNTS.index(k)
            out[k] = _first_crossing_year(hist[:, j], tgt_f, ppy)
        else:
            raise ValueError(f"Unknown target key: {k}")
    return out


def retirement_year_from_targets(
    sim: Dict[str, Any],
    targets: Dict[str, float],
    require: Optional[List[str]] = None,
) -> Optional[float]:
    yrs = crossing_years(sim, targets)
    req = require if require is not None else list(targets.keys())

    needed: List[float] = []
    for k in req:
        y = yrs.get(k, None)
        if y is None:
            return None
        needed.append(float(y))
    return max(needed) if needed else None


def yearly_snapshots(sim: Dict[str, Any]) -> pd.DataFrame:
    ppy = int(sim["periods_per_year"])
    N = int(sim["N_years"])
    hist = sim["history"]

    rows = []
    for y in range(1, N + 1):
        idx = y * ppy - 1
        bal = hist[idx, :]
        rows.append(
            {
                "Year": y,
                "IRA": float(bal[0]),
                "401K": float(bal[1]),
                "529": float(bal[2]),
                "ETF": float(bal[3]),
                "Total": float(bal.sum()),
            }
        )
    return pd.DataFrame(rows)


# ------------------------------
# Yahoo Finance DCA helper (choose any ticker, buy on 1st and/or 15th business day)
# ------------------------------

def fetch_yahoo_prices(
    ticker: str,
    start: Any,
    end: Any,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """
    Fetch daily price history via yfinance (Yahoo Finance).

    NOTE: Requires `yfinance` and an internet connection on your machine.
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError("Missing dependency: yfinance. Install with: pip install yfinance") from e

    df = yf.download(ticker, start=start, end=end, auto_adjust=auto_adjust, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No data returned for ticker {ticker!r}. Check the symbol and date range.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df.sort_index()


def _pick_nth_trading_day(dates: pd.DatetimeIndex, n: int) -> Optional[pd.Timestamp]:
    if len(dates) == 0:
        return None
    if n <= 1:
        return pd.Timestamp(dates[0])
    if len(dates) >= n:
        return pd.Timestamp(dates[n - 1])
    # If month has fewer than n trading days, fall back to last trading day
    return pd.Timestamp(dates[-1])


def simulate_yahoo_dca_monthly_values(
    ticker: str,
    start_date: Any,
    N_years: int,
    monthly_contribs: np.ndarray,
    buy_on: Iterable[str] = ("first", "15th"),
    price_col: str = "Close",
    auto_adjust: bool = True,
    invest_initial_balance: float = 0.0,
) -> Dict[str, Any]:
    """
    Backtest DCA into a Yahoo Finance ticker and return MONTH-END values aligned with the model's months.

    - monthly_contribs: length = N_years*12, dollars contributed each month (from phases)
    - buy_on: ("first"), ("15th"), or ("first","15th") business days of each month
      (interpreted as *trading days* in that month).
      If both are used, monthly contribution is split evenly across the two buys.
    - invest_initial_balance: if >0, invests this amount on the first buy date (or first trading day).

    Returns:
      - monthly_values: ndarray length n_months (month-end value; last trading day in month)
      - trades: DataFrame of purchases
      - daily: DataFrame with daily value
    """
    buy_on = tuple(buy_on)
    for b in buy_on:
        if b not in ("first", "15th"):
            raise ValueError("buy_on must be any of: 'first', '15th' (or both).")

    n_months = int(N_years * 12)
    monthly_contribs = np.asarray(monthly_contribs, dtype=float).reshape(-1)
    if len(monthly_contribs) != n_months:
        raise ValueError(f"monthly_contribs length must be {n_months}, got {len(monthly_contribs)}")

    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = (start_ts + pd.DateOffset(years=int(N_years))) + pd.Timedelta(days=7)

    prices = fetch_yahoo_prices(ticker, start=start_ts, end=end_ts, auto_adjust=auto_adjust)
    if price_col not in prices.columns:
        raise ValueError(f"price_col={price_col!r} not found. Available: {list(prices.columns)}")

    idx = prices.index

    # calendar months in horizon
    start_period = pd.Period(start_ts, freq="M")
    periods = [start_period + i for i in range(n_months)]

    # purchase plan (date, dollars)
    plan: List[Tuple[pd.Timestamp, float]] = []
    for i, per in enumerate(periods):
        amt = float(monthly_contribs[i])
        if amt <= 0:
            continue

        month_mask = (idx.year == per.year) & (idx.month == per.month)
        month_days = pd.DatetimeIndex(idx[month_mask]).sort_values()
        if len(month_days) == 0:
            continue

        first_day = _pick_nth_trading_day(month_days, 1)
        day_15 = _pick_nth_trading_day(month_days, 15)

        if buy_on == ("first",):
            plan.append((first_day, amt))
        elif buy_on == ("15th",):
            plan.append((day_15, amt))
        else:
            half = amt / 2.0
            plan.append((first_day, half))
            plan.append((day_15, half))

    plan = sorted(plan, key=lambda x: x[0])

    shares = 0.0
    trades_rows: List[Dict[str, Any]] = []

    if invest_initial_balance > 0:
        init_date = plan[0][0] if len(plan) > 0 else pd.Timestamp(idx[0])
        init_price = float(prices.loc[init_date, price_col])
        init_shares = float(invest_initial_balance) / init_price
        shares += init_shares
        trades_rows.append(
            {
                "date": pd.Timestamp(init_date),
                "dollars": float(invest_initial_balance),
                "price": init_price,
                "shares": init_shares,
                "cum_shares": shares,
                "note": "initial",
            }
        )

    for d, amt in plan:
        if d not in prices.index:
            pos = prices.index.searchsorted(d)
            if pos >= len(prices.index):
                continue
            d = prices.index[pos]
        px = float(prices.loc[d, price_col])
        sh = float(amt) / px
        shares += sh
        trades_rows.append(
            {"date": pd.Timestamp(d), "dollars": float(amt), "price": px, "shares": sh, "cum_shares": shares, "note": ""}
        )

    trades = (
        pd.DataFrame(trades_rows).sort_values("date")
        if trades_rows
        else pd.DataFrame(columns=["date", "dollars", "price", "shares", "cum_shares", "note"])
    )

    # Daily value
    daily = prices[[price_col]].copy()
    if not trades.empty:
        s = pd.Series(index=prices.index, data=np.nan)
        for _, r in trades.iterrows():
            s.loc[pd.Timestamp(r["date"])] = float(r["cum_shares"])
        daily["cum_shares"] = s.ffill().fillna(0.0)
    else:
        daily["cum_shares"] = 0.0
    daily["value"] = daily["cum_shares"] * daily[price_col]

    # Month-end values aligned to periods (last trading day in each month)
    monthly_values = np.zeros(n_months, dtype=float)
    for i, per in enumerate(periods):
        month_mask = (daily.index.year == per.year) & (daily.index.month == per.month)
        month_days = daily.loc[month_mask]
        if month_days.empty:
            monthly_values[i] = monthly_values[i - 1] if i > 0 else float(invest_initial_balance)
        else:
            monthly_values[i] = float(month_days["value"].iloc[-1])

    return {
        "ticker": ticker,
        "start_used": str(prices.index.min().date()),
        "end_used": str(prices.index.max().date()),
        "prices": prices,
        "daily": daily,
        "trades": trades,
        "monthly_values": monthly_values,
    }


# ------------------------------
# PDF Report Builder (reportlab)
# ------------------------------

def build_report_pdf_bytes(
    sim: Dict[str, Any],
    targets: Dict[str, float],
    crossing: Dict[str, Optional[float]],
    retire_year: Optional[float],
    require: Optional[List[str]],
    fig: Optional[plt.Figure] = None,
    market_mode: Optional[Dict[str, Any]] = None,
) -> bytes:
    """
    Full report PDF bytes:
      - inputs
      - phases
      - targets + crossing years
      - summary (final balances / contrib / growth)
      - chart
      - year-end snapshots
    """
    from io import BytesIO
    from datetime import datetime

    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.lib.units import inch

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        title="Retirement Hunt Report",
    )

    styles = getSampleStyleSheet()
    h1 = styles["Heading1"]
    h2 = styles["Heading2"]
    h3 = styles["Heading3"]
    body = styles["BodyText"]

    def money(x: Any) -> str:
        try:
            return f"${float(x):,.0f}"
        except Exception:
            return str(x)

    def yearfmt(x: Any) -> str:
        if x is None:
            return "Not reached"
        return f"{float(x):.2f}"

    def make_table(data, colWidths=None, header_rows=1, font_size=9):
        t = Table(data, colWidths=colWidths, repeatRows=header_rows)
        t.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), font_size),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAEAEA")),
                    ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                    ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#BBBBBB")),
                ]
            )
        )
        return t

    story = []
    story.append(Paragraph("Retirement Hunt Report", h1))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body))
    story.append(Spacer(1, 10))

    # Inputs
    story.append(Paragraph("Inputs", h2))
    sb = sim["start_balances"]
    rates = sim["annual_rates_pct"]
    inputs_tbl = [
        ["Item", "IRA", "401K", "529", "ETF"],
        ["Starting Balances", money(sb["IRA"]), money(sb["401K"]), money(sb["529"]), money(sb["ETF"])],
        ["Annual Rates (%)", f"{rates['IRA']:.2f}%", f"{rates['401K']:.2f}%", f"{rates['529']:.2f}%", f"{rates['ETF']:.2f}%"],
    ]
    story.append(make_table(inputs_tbl))

    # Optional: Market Mode sections (Yahoo Finance) for one or more accounts
    # `market_mode` can be either:
    #   - a single dict describing one account (legacy; assumed ETF), or
    #   - a dict mapping account -> dict(details)
    market_modes = None
    if market_mode:
        try:
            # If it looks like a single config (has 'ticker' field), treat as ETF config
            if isinstance(market_mode, dict) and ("ticker" in market_mode or "trades" in market_mode):
                market_modes = {"ETF": market_mode}
            elif isinstance(market_mode, dict):
                # If keys are accounts, treat as multi
                if any(k in ACCOUNTS for k in market_mode.keys()):
                    market_modes = market_mode
        except Exception:
            market_modes = None

    if market_modes:
        story.append(Paragraph("Market Mode (Yahoo Finance)", h2))

        for acct in ["IRA", "401K", "529", "ETF"]:
            cfg = market_modes.get(acct)
            if not cfg:
                continue
            if not bool(cfg.get("enabled", True)):
                continue

            story.append(Paragraph(f"{acct} Market Mode", h3))

            mm_tbl = [
                ["Field", "Value"],
                ["Ticker", str(cfg.get("ticker", ""))],
                ["Data range", f"{cfg.get('start_used','')} to {cfg.get('end_used','')}"],
                ["Buy on", str(cfg.get("buy_on", ""))],
                ["Invest initial balance", str(cfg.get("invest_initial", ""))],
            ]
            story.append(make_table(mm_tbl, colWidths=[2.0*inch, 5.0*inch], header_rows=1, font_size=9))
            story.append(Spacer(1, 8))

            trades = cfg.get("trades")
            has_trades = False
            try:
                has_trades = trades is not None and hasattr(trades, "empty") and (not trades.empty)
            except Exception:
                has_trades = False

            if has_trades:
                # Summary
                total_dollars = float(trades["dollars"].sum()) if "dollars" in trades.columns else 0.0
                total_shares = float(trades["shares"].sum()) if "shares" in trades.columns else 0.0
                first_date = str(trades["date"].min()) if "date" in trades.columns else ""
                last_date = str(trades["date"].max()) if "date" in trades.columns else ""

                story.append(Paragraph("Trades summary", body))
                summary_tbl = [
                    ["Metric", "Value"],
                    ["Total invested", money(total_dollars)],
                    ["Total shares purchased", f"{total_shares:,.6f}"],
                    ["First trade", first_date],
                    ["Last trade", last_date],
                ]
                story.append(make_table(summary_tbl, colWidths=[2.0*inch, 5.0*inch], header_rows=1, font_size=9))
                story.append(Spacer(1, 8))

                # Full trades table (paginated via splitting into chunks)
                cols = [c for c in ["date", "dollars", "price", "shares", "cum_shares", "note"] if c in trades.columns]
                show = trades[cols].copy()

                if "date" in show.columns:
                    show["date"] = show["date"].astype(str)

                for c in ["dollars", "price", "shares", "cum_shares"]:
                    if c in show.columns:
                        show[c] = show[c].map(lambda x: x if pd.isna(x) else float(x))

                def fmt_row(row):
                    out = []
                    for c in cols:
                        v = row[c]
                        if c == "dollars":
                            out.append(money(v))
                        elif c == "price":
                            out.append(f"{float(v):,.4f}" if v == v else "")
                        elif c in ("shares", "cum_shares"):
                            out.append(f"{float(v):,.6f}" if v == v else "")
                        else:
                            out.append(str(v) if v is not None else "")
                    return out

                data_rows = [cols] + [fmt_row(r) for _, r in show.iterrows()]
                chunk_size = 35  # rows per page chunk (incl header)
                start_i = 1
                while start_i < len(data_rows):
                    chunk = [data_rows[0]] + data_rows[start_i:start_i + chunk_size]
                    story.append(Paragraph(f"Trades (continued) — {acct}", body))
                    story.append(make_table(chunk, header_rows=1, font_size=8))
                    story.append(Spacer(1, 8))
                    start_i += chunk_size
            else:
                story.append(Paragraph("No trades were generated (likely $0 contributions).", body))
                story.append(Spacer(1, 8))

        story.append(Spacer(1, 12))

    # Targets
    story.append(Paragraph("Targets and Crossing Years", h2))
    t_tbl = [["Account", "Target", "Crossing Year"]]
    for k, v in targets.items():
        t_tbl.append([k, money(v), yearfmt(crossing.get(k))])
    story.append(make_table(t_tbl))
    story.append(Spacer(1, 8))

    req = require if require else list(targets.keys())
    story.append(Paragraph(f"Retirement criteria: required targets = {', '.join(req)}", body))
    story.append(Paragraph(f"Retirement year: {('Not reached' if retire_year is None else f'{retire_year:.2f}')}", body))
    story.append(PageBreak())

    # Summary
    story.append(Paragraph("Summary", h2))
    fr = sim["final_report"]
    fb = fr["Final Balances"]
    tc = fr["Total Contributions"]
    gr = fr["Growth"]
    summary_tbl = [
        ["Metric", "IRA", "401K", "529", "ETF", "Total"],
        ["Final Balances", money(fb["IRA"]), money(fb["401K"]), money(fb["529"]), money(fb["ETF"]), money(fb["Total"])],
        ["Total Contributions", money(tc["IRA"]), money(tc["401K"]), money(tc["529"]), money(tc["ETF"]), money(tc["Total"])],
        ["Growth", money(gr["IRA"]), money(gr["401K"]), money(gr["529"]), money(gr["ETF"]), money(gr["Total"])],
    ]
    story.append(make_table(summary_tbl, font_size=8))
    story.append(PageBreak())

    # Chart
    story.append(Paragraph("Growth Chart", h2))
    if fig is None:
        fig = make_growth_figure(sim, embed_summary=False)

    from io import BytesIO as _BytesIO
    img_buf = _BytesIO()
    fig.savefig(img_buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    img_buf.seek(0)

    img = Image(img_buf)
    img.hAlign = 'CENTER'
    # Scale image to fit within the printable frame to avoid LayoutError
    img._restrictSize(doc.width, doc.height)
    story.append(img)
    story.append(PageBreak())

    # Year-end snapshots
    story.append(Paragraph("Year-End Snapshots", h2))
    ys = yearly_snapshots(sim)
    header = ["Year", "IRA", "401K", "529", "ETF", "Total"]
    rows = [[int(r["Year"]), money(r["IRA"]), money(r["401K"]), money(r["529"]), money(r["ETF"]), money(r["Total"])] for _, r in ys.iterrows()]

    rows_per_page = 35
    for i in range(0, len(rows), rows_per_page):
        chunk = rows[i:i + rows_per_page]
        story.append(make_table([header] + chunk, font_size=7))
        if i + rows_per_page < len(rows):
            story.append(PageBreak())

    doc.build(story)
    return buf.getvalue()

# ------------------------------
# Goal Solver: required extra monthly contribution to hit a target by a given year
# ------------------------------

def _value_at_month(sim: Dict[str, Any], month_idx: int, key: str) -> float:
    """key in ACCOUNTS or 'Total'"""
    month_idx = int(month_idx)
    if month_idx < 0:
        raise ValueError("month_idx must be >= 0")
    if key == "Total":
        return float(sim["total"][month_idx])
    if key not in ACCOUNTS:
        raise ValueError(f"Unknown key: {key}. Use one of {ACCOUNTS} or 'Total'.")
    j = ACCOUNTS.index(key)
    return float(sim["history"][month_idx, j])


def _add_extra_to_phases(phases: List[Phase], account: str, extra_monthly: float) -> List[Phase]:
    """Return a new phases list where the selected account gets +extra_monthly in every phase."""
    extra_monthly = float(extra_monthly)
    out: List[Phase] = []
    for p in phases:
        if account == "IRA":
            out.append(Phase(p.phase, p.start_year, p.end_year, p.ira + extra_monthly, p.tsp, p.college, p.etf))
        elif account == "401K":
            out.append(Phase(p.phase, p.start_year, p.end_year, p.ira, p.tsp + extra_monthly, p.college, p.etf))
        elif account == "529":
            out.append(Phase(p.phase, p.start_year, p.end_year, p.ira, p.tsp, p.college + extra_monthly, p.etf))
        elif account == "ETF":
            out.append(Phase(p.phase, p.start_year, p.end_year, p.ira, p.tsp, p.college, p.etf + extra_monthly))
        else:
            raise ValueError("account must be one of: IRA, 401K, 529, ETF")
    return out


def solve_extra_monthly_contribution(
    *,
    start_balances: Dict[str, float],
    annual_rates_pct: Dict[str, float],
    phases: List[Phase],
    N_years: int,
    target_key: str,
    target_value: float,
    goal_year: int,
    extra_to: Optional[str] = None,
    periods_per_year: int = 12,
    tol: float = 1.0,
    max_iter: int = 40,
) -> Dict[str, Any]:
    """
    Solve for the ADDITIONAL monthly contribution needed to reach a target by goal_year.

    - target_key: one of 'IRA','401K','529','ETF','Total' (what you want to hit)
    - extra_to: which account receives the additional contribution. If target_key != 'Total',
      extra_to defaults to target_key. If target_key == 'Total', extra_to must be set (e.g. 'ETF').

    Returns dict with:
      baseline_value, shortfall, extra_monthly_required, achieved_value, goal_year
    """
    if goal_year <= 0:
        raise ValueError("goal_year must be >= 1")
    if goal_year > int(N_years):
        raise ValueError("goal_year cannot exceed N_years")

    target_key = str(target_key)
    target_value = float(target_value)

    if target_key == "Total":
        if extra_to is None:
            raise ValueError("For target_key='Total', you must provide extra_to (IRA/401K/529/ETF).")
        extra_to = str(extra_to)
        if extra_to not in ACCOUNTS:
            raise ValueError(f"extra_to must be one of {ACCOUNTS}")
    else:
        if target_key not in ACCOUNTS:
            raise ValueError(f"target_key must be one of {ACCOUNTS} or 'Total'")
        extra_to = target_key

    # baseline
    base = simulate_accounts(
        start_balances=start_balances,
        annual_rates_pct=annual_rates_pct,
        N_years=int(N_years),
        phases=phases,
        periods_per_year=int(periods_per_year),
    )
    goal_month_idx = goal_year * int(periods_per_year) - 1
    baseline_value = _value_at_month(base, goal_month_idx, target_key)

    if baseline_value >= target_value:
        return {
            "goal_year": int(goal_year),
            "target_key": target_key,
            "target_value": target_value,
            "extra_to": extra_to,
            "baseline_value": baseline_value,
            "shortfall": 0.0,
            "extra_monthly_required": 0.0,
            "achieved_value": baseline_value,
            "iterations": 0,
        }

    shortfall0 = target_value - baseline_value

    # Find high bound
    low = 0.0
    high = max(10.0, shortfall0 / max(1, goal_year * 12) * 2.0)

    def eval_with(extra: float) -> float:
        ph2 = _add_extra_to_phases(phases, extra_to, extra)
        sim2 = simulate_accounts(
            start_balances=start_balances,
            annual_rates_pct=annual_rates_pct,
            N_years=int(N_years),
            phases=ph2,
            periods_per_year=int(periods_per_year),
        )
        return _value_at_month(sim2, goal_month_idx, target_key)

    val_high = eval_with(high)
    guard = 0
    while val_high < target_value and guard < 30:
        high *= 2.0
        val_high = eval_with(high)
        guard += 1
        if high > 1e7:
            break

    if val_high < target_value:
        return {
            "goal_year": int(goal_year),
            "target_key": target_key,
            "target_value": target_value,
            "extra_to": extra_to,
            "baseline_value": baseline_value,
            "shortfall": shortfall0,
            "extra_monthly_required": None,
            "achieved_value": val_high,
            "iterations": guard,
            "note": "Could not bracket a solution. Try increasing N_years or reducing target.",
        }

    # Binary search
    it = 0
    best = high
    best_val = val_high

    for it in range(1, int(max_iter) + 1):
        mid = (low + high) / 2.0
        v = eval_with(mid)

        if v >= target_value:
            best = mid
            best_val = v
            high = mid
        else:
            low = mid

        if abs(v - target_value) <= float(tol):
            best = mid
            best_val = v
            break

    return {
        "goal_year": int(goal_year),
        "target_key": target_key,
        "target_value": target_value,
        "extra_to": extra_to,
        "baseline_value": baseline_value,
        "shortfall": max(0.0, target_value - baseline_value),
        "extra_monthly_required": float(best),
        "achieved_value": float(best_val),
        "iterations": int(it),
        "tolerance": float(tol),
    }

# ------------------------------
# Goal Solver (v2): extra monthly contribution applied only within a year window
# ------------------------------

def _add_extra_to_phases_window(
    phases: List[Phase],
    account: str,
    extra_monthly: float,
    start_year: int,
    end_year: int,
) -> List[Phase]:
    """Return a new phases list where `account` gets +extra_monthly only for years [start_year, end_year)."""
    extra_monthly = float(extra_monthly)
    start_year = int(start_year)
    end_year = int(end_year)
    if end_year <= start_year:
        return list(phases)

    out: List[Phase] = []
    for p in phases:
        s = int(p.start_year)
        e = int(p.end_year)
        # no overlap
        if e <= start_year or s >= end_year:
            out.append(p)
            continue

        # segments: pre, mid, post
        pre_s, pre_e = s, min(e, start_year)
        mid_s, mid_e = max(s, start_year), min(e, end_year)
        post_s, post_e = max(s, end_year), e

        def mk(seg_s, seg_e, add: bool) -> Phase:
            if account == "IRA":
                return Phase(p.phase, seg_s, seg_e, p.ira + (extra_monthly if add else 0.0), p.tsp, p.college, p.etf)
            if account == "401K":
                return Phase(p.phase, seg_s, seg_e, p.ira, p.tsp + (extra_monthly if add else 0.0), p.college, p.etf)
            if account == "529":
                return Phase(p.phase, seg_s, seg_e, p.ira, p.tsp, p.college + (extra_monthly if add else 0.0), p.etf)
            if account == "ETF":
                return Phase(p.phase, seg_s, seg_e, p.ira, p.tsp, p.college, p.etf + (extra_monthly if add else 0.0))
            raise ValueError("account must be one of: IRA, 401K, 529, ETF")

        if pre_e > pre_s:
            out.append(mk(pre_s, pre_e, add=False))
        if mid_e > mid_s:
            out.append(mk(mid_s, mid_e, add=True))
        if post_e > post_s:
            out.append(mk(post_s, post_e, add=False))

    # sort by (start,end,phase) to keep deterministic ordering
    out.sort(key=lambda x: (int(x.start_year), int(x.end_year), int(x.phase)))
    return out


def solve_extra_monthly_contribution(
    *,
    start_balances: Dict[str, float],
    annual_rates_pct: Dict[str, float],
    phases: List[Phase],
    N_years: int,
    target_key: str,
    target_value: float,
    goal_year: int,
    extra_to: str,
    extra_start_year: int = 0,
    extra_end_year: Optional[int] = None,
    periods_per_year: int = 12,
    tol: float = 5.0,
    max_iter: int = 40,
) -> Dict[str, Any]:
    """
    Solve for the ADDITIONAL monthly contribution needed to reach a target by `goal_year`,
    applying that extra contribution only from `extra_start_year` (inclusive) through `extra_end_year` (exclusive).
    If extra_end_year is None, it defaults to goal_year.

    - target_key: 'IRA','401K','529','ETF','Total' (what you want to hit)
    - extra_to: which account receives the extra monthly contribution
    """
    goal_year = int(goal_year)
    extra_start_year = int(extra_start_year)
    if extra_end_year is None:
        extra_end_year = goal_year
    extra_end_year = int(extra_end_year)

    if goal_year <= 0:
        raise ValueError("goal_year must be >= 1")
    if goal_year > int(N_years):
        raise ValueError("goal_year cannot exceed N_years")
    if extra_start_year < 0:
        raise ValueError("extra_start_year must be >= 0")
    if extra_start_year >= goal_year:
        raise ValueError("extra_start_year must be < goal_year to affect the result")
    if extra_end_year <= extra_start_year:
        raise ValueError("extra_end_year must be > extra_start_year")
    if extra_to not in ACCOUNTS:
        raise ValueError(f"extra_to must be one of {ACCOUNTS}")

    target_key = str(target_key)
    target_value = float(target_value)

    if target_key != "Total" and target_key not in ACCOUNTS:
        raise ValueError(f"target_key must be one of {ACCOUNTS} or 'Total'")

    # baseline
    base = simulate_accounts(
        start_balances=start_balances,
        annual_rates_pct=annual_rates_pct,
        N_years=int(N_years),
        phases=phases,
        periods_per_year=int(periods_per_year),
    )
    goal_month_idx = goal_year * int(periods_per_year) - 1
    baseline_value = _value_at_month(base, goal_month_idx, target_key)

    if baseline_value >= target_value:
        return {
            "goal_year": int(goal_year),
            "target_key": target_key,
            "target_value": target_value,
            "extra_to": extra_to,
            "extra_start_year": int(extra_start_year),
            "extra_end_year": int(extra_end_year),
            "baseline_value": baseline_value,
            "shortfall": 0.0,
            "extra_monthly_required": 0.0,
            "achieved_value": baseline_value,
            "iterations": 0,
        }

    shortfall0 = target_value - baseline_value

    # Find high bound
    low = 0.0
    # heuristic: distribute shortfall over remaining months with some cushion
    months_active = max(1, (extra_end_year - extra_start_year) * int(periods_per_year))
    high = max(10.0, (shortfall0 / months_active) * 2.0)

    def eval_with(extra: float) -> float:
        ph2 = _add_extra_to_phases_window(phases, extra_to, extra, extra_start_year, extra_end_year)
        sim2 = simulate_accounts(
            start_balances=start_balances,
            annual_rates_pct=annual_rates_pct,
            N_years=int(N_years),
            phases=ph2,
            periods_per_year=int(periods_per_year),
        )
        return _value_at_month(sim2, goal_month_idx, target_key)

    val_high = eval_with(high)
    guard = 0
    while val_high < target_value and guard < 30:
        high *= 2.0
        val_high = eval_with(high)
        guard += 1
        if high > 1e7:
            break

    if val_high < target_value:
        return {
            "goal_year": int(goal_year),
            "target_key": target_key,
            "target_value": target_value,
            "extra_to": extra_to,
            "extra_start_year": int(extra_start_year),
            "extra_end_year": int(extra_end_year),
            "baseline_value": baseline_value,
            "shortfall": shortfall0,
            "extra_monthly_required": None,
            "achieved_value": val_high,
            "iterations": guard,
            "note": "Could not bracket a solution. Try increasing horizon or reducing target.",
        }

    # Binary search
    best = high
    best_val = val_high

    it = 0
    for it in range(1, int(max_iter) + 1):
        mid = (low + high) / 2.0
        v = eval_with(mid)

        if v >= target_value:
            best = mid
            best_val = v
            high = mid
        else:
            low = mid

        if abs(v - target_value) <= float(tol):
            best = mid
            best_val = v
            break

    return {
        "goal_year": int(goal_year),
        "target_key": target_key,
        "target_value": target_value,
        "extra_to": extra_to,
        "extra_start_year": int(extra_start_year),
        "extra_end_year": int(extra_end_year),
        "baseline_value": baseline_value,
        "shortfall": max(0.0, target_value - baseline_value),
        "extra_monthly_required": float(best),
        "achieved_value": float(best_val),
        "iterations": int(it),
        "tolerance": float(tol),
    }
