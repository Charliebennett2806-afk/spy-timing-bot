# SPY Timing Bot
Daily S&P 500 (SPY) timing strategy using walk-forward logistic regression and regime logic.

## Quick start
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 timing_bot.py     # default: since 2005 + plots (+ multi panel)

## Other runs
python3 timing_bot.py --multi --outdir outputs_multi
python3 timing_bot.py --compare --outdir outputs_cmp
python3 timing_bot.py --years 5 --outdir outputs_5y

Research code only — not investment advice.