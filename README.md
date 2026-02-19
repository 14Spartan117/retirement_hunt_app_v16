# Retirement Hunt (Streamlit)

This app:
1) Simulates IRA / 401K / 529 / ETF balances with monthly compounding and phase-based monthly contributions.
2) "Hunts" for the first year each account crosses a target value.
3) Computes a "retirement year" = max crossing year among required targets.

## Files
- `finance_model.py`  -> simulation + plotting + crossing-year utilities
- `app.py`            -> Streamlit UI
- `requirements.txt`  -> dependencies

## Setup
```bash
python -m venv .venv
source .venv/bin/activate          # mac/linux
# .venv\Scripts\activate         # windows

pip install -r requirements.txt
```

## Run
```bash
streamlit run app.py
```

## Notes
- Phase `end_year` is **exclusive** (Phase end_year=10 runs through year 9.x).
- If a target isn't reached within the horizon, its crossing year is blank and retirement will fail.


## v5: Yahoo Finance “Buy on 1st / 15th business day” (for ETF)

In the sidebar, turn on **ETF Market Mode (Yahoo Finance)**:
- Pick a ticker (Yahoo symbol, e.g. SPY, QQQ, AAPL, META)
- Choose to buy on the **1st and/or 15th trading day** of each month
- The ETF account is overridden using historical prices (backtest)

Notes:
- If you pick both days, the app splits your *monthly ETF contribution* across the two buys.
- This requires internet access and `yfinance` (included in requirements.txt).


## v6: Trades table + CSV download
The app shows an expandable panel with the Yahoo Finance trade log, and you can download trades as CSV.
The full PDF report also includes market-mode details and a paginated trades table.


## v7: PDF chart scaling fix
Fixed ReportLab LayoutError by auto-scaling embedded chart images to fit the printable frame.


## v8: Fixed Streamlit indentation + start_balances NameError
Rebuilt app.py cleanly so market mode, plotting, and downloads only run after pressing **Run Hunt**.


## v9: Results persist after Run
Outputs now persist using `st.session_state`, so charts/tables/downloads don't disappear after you interact with inputs.


## v10: Comma formatting + Goal Solver
- Final Report and Crossing Years now show comma-formatted currency.
- Added a Goal Solver panel that estimates the extra monthly contribution needed to hit a target by a chosen year.
- Escaped dollar signs in chart text so the embedded summary renders correctly.


## v11: Solver start year + per-account Market Mode
- Goal Solver now supports choosing a start year and end year window for extra contributions.
- Any account (IRA/401K/529/ETF) can be modeled as a Yahoo Finance ticker with 1st/15th trading-day buys.
- PDF report includes Market Mode sections for all enabled accounts.


## v12: Excel tabs (Budget + Conscious Spending Plan)
- Added two tabs that load and edit the uploaded Excel templates.
- You can scale selected rows/sections by a % and push extracted values into Phase 1 contributions.
- Spreadsheets are included under ./data by default (you can also upload newer versions in-app).


## v15: Baseline sheet is always editable + build label
- FY26 Baseline tab now shows an always-editable data editor for 'Stacking Bread'.
- Added Build label at top + sidebar so you can confirm which zip you’re running.
