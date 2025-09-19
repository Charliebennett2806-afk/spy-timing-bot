#!/usr/bin/env python3
"""
SPY Timing Bot (daily) — backtest & live signal (robust version)

Run:
  # Default (no flags): full backtest since 2005 with plots
  python3 timing_bot.py

  # Examples:
  python3 timing_bot.py --years 5 --outdir outputs_5y
  python3 timing_bot.py --start 2005-01-01 --outdir outputs_full
  python3 timing_bot.py --years 5 --live
  python3 timing_bot.py --multi --outdir outputs_multi
  python3 timing_bot.py --compare --outdir outputs_cmp
"""

import os, sys, json, argparse, warnings
import datetime
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import yfinance as yf

__version__ = "1.0.0"

# ---- unified thresholds ----

P_BUY = 0.505     # long if p_up >= P_BUY
P_SELL = 0.495    # short if p_up <= P_SELL
P_LONG_LIVE = P_BUY   # live mode uses same by default
P_SHORT_LIVE = P_SELL

# --- regime config (single source of truth) ---
BULL_FLOOR = 0.40               # min long when MA50 > MA200
LONG_CAP_NO_CONTANGO = 0.50     # cap long size when contango is OFF


warnings.filterwarnings("ignore", category=FutureWarning)

SECTORS = ["XLF","XLK","XLE","XLI","XLV","XLY","XLP","XLU","XLB","XLRE"]

# ---------- helpers ----------
def _cols_to_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Force single-level string column names (flatten tuples)."""
    def _name(c):
        if isinstance(c, tuple):
            return "_".join([str(x) for x in c if x is not None])
        return str(c)
    df = df.copy()
    df.columns = [_name(c) for c in df.columns]
    return df

# ---------- Data loading (robust to yfinance quirks) ----------
def fetch_data(start=None, years=5):
    # choose date
    if start is None:
        end = pd.Timestamp.today().normalize()
        start = (end - pd.DateOffset(years=years)).strftime("%Y-%m-%d")

    # SPY adjusted OHLCV (use auto_adjust=True so Close is split/dividend adjusted)
    spy = yf.download("SPY", start=start, auto_adjust=True, progress=False)
    if spy is None or spy.empty:
        raise RuntimeError("Failed to download SPY data.")
    spy = spy[["Open","High","Low","Close","Volume"]].copy()
    spy["PX_CLOSE"] = spy["Close"]  # keep both 'Close' and 'PX_CLOSE'

    # VIX and VIX3M (VXV fallback) — get Close series and name them
    def close_series(ticker, name):
        df = yf.download(ticker, start=start, auto_adjust=False, progress=False)
        if df is None or df.empty:
            return pd.Series(dtype=float, name=name)
        if isinstance(df.columns, pd.MultiIndex):
            # ('Close','TICK') or ('TICK','Close')
            if "Close" in df.columns.get_level_values(0):
                s = df.xs("Close", level=0, axis=1).squeeze()
            else:
                s = df.xs("Close", level=1, axis=1).squeeze()
            if isinstance(s, pd.DataFrame) and s.shape[1] == 1:
                s = s.iloc[:,0]
        else:
            s = df["Close"] if "Close" in df.columns else df.iloc[:,0]
        return pd.Series(s, name=name)

    vix  = close_series("^VIX", "VIX")
    vix3 = close_series("^VIX3M", "VIX3M")
    if vix3.dropna().empty:
        vix3 = close_series("^VXV", "VIX3M")
    # Final safety: if VIX3M/VXV is missing early in history, fall back to VIX so vix_slope is defined
    vix3 = vix3.combine_first(vix)

    # Sector ETF closes (adjusted), flattened to single-level columns
    sec_raw = yf.download(SECTORS, start=start, auto_adjust=True, progress=False)
    if sec_raw is None or sec_raw.empty:
        raise RuntimeError("Failed to download sector ETF data.")
    if isinstance(sec_raw.columns, pd.MultiIndex):
        if "Close" in sec_raw.columns.get_level_values(0):
            sec = sec_raw.xs("Close", level=0, axis=1)
        else:
            sec = sec_raw.xs("Close", level=1, axis=1)
    else:
        sec = sec_raw["Close"] if "Close" in sec_raw.columns else sec_raw
    if isinstance(sec, pd.Series):
        sec = sec.to_frame()
    sec.columns = [f"SEC_{str(c)}" for c in sec.columns]

    # Assemble (be lenient so long histories aren't truncated by newer columns like XLRE)
    data = pd.concat([spy, vix.to_frame(), vix3.to_frame(), sec], axis=1)
    data = _cols_to_strings(data).sort_index()
    # Drop columns that are entirely missing (e.g., ETFs that didn't exist yet)
    data = data.dropna(axis=1, how="all")
    # Only require the core columns to exist on each row; let optional inputs be NaN
    required = []
    if "PX_CLOSE" in data.columns:
        required.append("PX_CLOSE")
    elif "Close" in data.columns:
        required.append("Close")
    if "VIX" in data.columns:
        required.append("VIX")
    if required:
        data = data.dropna(subset=required)
    # Forward-fill optional columns a bit to avoid spurious drops from isolated gaps
    data = data.ffill(limit=5)
    return data

# ---------- Features ----------
def build_features(data: pd.DataFrame):
    df = data.copy()

    # choose a safe close column (simple + resilient)
    if "PX_CLOSE" in df.columns and df["PX_CLOSE"].notna().any():
        close = df["PX_CLOSE"]
    elif "Close" in df.columns and df["Close"].notna().any():
        close = df["Close"]
    else:
        # last resort: first numeric column
        num = df.select_dtypes(include=[np.number])
        if num.shape[1] == 0:
            raise RuntimeError("No numeric columns to compute returns from.")
        close = num.iloc[:, 0]

    # make sure it's a Series
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce")

    # daily returns
    df["ret_cc"] = close.pct_change()



    # trend (regime)
    df["ma50"]  = close.rolling(50).mean()
    df["ma200"] = close.rolling(200).mean()
    df["trend_up"] = (df["ma50"] > df["ma200"]).astype(int)

    # mean reversion
    df["r1"] = df["ret_cc"]
    df["r2"] = df["ret_cc"].rolling(2).sum()
    df["r5"] = df["ret_cc"].rolling(5).sum()
    df["dist_ma10"] = close / close.rolling(10).mean() - 1.0

    # RSI(2)
    chg = close.diff()
    gain = chg.clip(lower=0).rolling(2).mean()
    loss = -chg.clip(upper=0).rolling(2).mean()
    rs = gain / (loss.replace(0, np.nan))
    df["rsi2"] = 100 - 100/(1+rs)

    # volatility regime
    df["vix_lvl"] = df["VIX"]
    # If VIX3M is missing, treat slope as 0 (flat term-structure)
    vix3m_safe = df["VIX3M"].combine_first(df["VIX"])
    df["vix_slope"] = (vix3m_safe / df["VIX"] - 1.0)
    df["vix_contango"] = (df["vix_slope"] > 0).astype(int)

    # breadth across sectors
    sec_cols = [c for c in df.columns if str(c).startswith("SEC_")]
    sec_ret = df[sec_cols].pct_change()
    df["breadth_ret1"] = sec_ret.mean(axis=1)
    above20 = (df[sec_cols] / df[sec_cols].rolling(20).mean() - 1.0) > 0
    df["breadth_above20"] = above20.sum(axis=1) / above20.shape[1]

    # target label: next-day up?
    df["y"] = (df["ret_cc"].shift(-1) > 0).astype(int)

    features = [
        "r1","r2","r5","dist_ma10","rsi2",
        "vix_lvl","vix_slope","vix_contango",
        "breadth_ret1","breadth_above20"
    ]
    # Be tolerant: require core features that we always have; allow occasional NaNs in breadth/vix_slope early on
    core_feats = ["r1","r2","r5","dist_ma10","rsi2","vix_lvl","breadth_ret1","breadth_above20","y","ret_cc","trend_up"]
    existing_core = [c for c in core_feats if c in df.columns]
    df = df.dropna(subset=existing_core).copy()
    return df, features

# ---------- Walk-forward model ----------
def walk_forward_probs(df: pd.DataFrame, features, initial_train_days=750) -> pd.Series:
    if len(df) <= initial_train_days + 1:
        raise RuntimeError("Not enough history for the chosen initial training window.")
    scaler = StandardScaler()
    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    prob = pd.Series(index=df.index, dtype=float)
    for t in range(initial_train_days, len(df)-1):
        train = df.iloc[:t]
        Xtr = scaler.fit_transform(train[features]); ytr = train["y"].values
        model.fit(Xtr, ytr)
        Xte = scaler.transform(df.iloc[[t]][features])
        prob.iloc[t] = model.predict_proba(Xte)[0,1]
    return prob.ffill()

# ---------- From probabilities to returns ----------
def apply_strategy(
    df: pd.DataFrame,
    p_up: pd.Series,
    # --- bull/bear regime parameters ---
    min_long_floor: float = BULL_FLOOR,     # floor long exposure when MA50 > MA200
    p_buy_bull: float  = 0.51,              # easier long threshold in bull regime
    p_sell_bull: float = 0.49,              # symmetric short threshold in bull (usually unused)
    p_buy_bear: float  = 0.53,              # stricter long threshold in bear regime
    p_sell_bear: float = 0.47,              # short threshold in bear regime
    # --- market structure guard ---
    long_cap_no_contango: float = LONG_CAP_NO_CONTANGO,  # cap long size when contango is OFF
    # --- hysteresis in bull regime (enter > exit to avoid flip-flop) ---
    bull_enter: float = 0.53,   # need ≥ this to scale above the floor in bulls
    bull_exit:  float = 0.50,   # fall back to floor only if ≤ this in bulls
    # --- execution cost ---
    cost_per_turn: float = 0.0002
):
    """
    Bull-patch:
      - Raise baseline long exposure in bull regimes (MA50>MA200) via min_long_floor.
      - Use regime-aware probability thresholds (easier to go long in bulls).
      - Keep existing bear protections: shorts only when MA50<=MA200; soft contango cap.
      - No leverage/vol-targeting/smoothing here.
    """
    out = df.copy()
    out["p_up"] = p_up.reindex(out.index).ffill()

    # Regime flags
    trend_up = (out["ma50"] > out["ma200"]).astype(int)   # 1 in bull, 0 in bear
    contango = out["vix_contango"].astype(int)            # 1 if VIX3M>VIX (calm)

    # Regime-aware thresholds per day
    p_buy  = np.where(trend_up == 1, p_buy_bull,  p_buy_bear)
    p_sell = np.where(trend_up == 1, p_sell_bull, p_sell_bear)

    # Build signal with hysteresis in bull regime (stateful)
    sig = np.zeros(len(out), dtype=float)
    prev = 0.0  # previous day's position

    pu_arr       = out["p_up"].values
    bull_arr     = trend_up.values
    contango_arr = contango.values

    # Ensure the band is sensible
    enter_thr = max(bull_enter, p_buy_bull)   # enter needs to be at least buy threshold
    exit_thr  = min(bull_exit,  p_sell_bull)  # exit no higher than sell threshold

    for i in range(len(out)):
        pu   = pu_arr[i]
        bull = (bull_arr[i] == 1)
        calm = (contango_arr[i] == 1)  # contango ON

        if bull:
            # baseline: floor in bulls
            size = min_long_floor

            if prev <= min_long_floor:
                # only scale up above floor when confidence crosses enter band
                if pu >= enter_thr:
                    size = 1.0 if calm else min(long_cap_no_contango, 1.0)
            else:
                # already > floor: only drop back to floor if we cross the exit band
                if pu <= exit_thr:
                    size = min_long_floor
                else:
                    size = 1.0 if calm else min(long_cap_no_contango, 1.0)

        else:
            # Bear regime: keep your existing behavior
            if pu >= p_buy_bear:
                size = 1.0 if calm else min(long_cap_no_contango, 1.0)
            elif pu <= p_sell_bear:
                size = -1.0
            else:
                size = 0.0

        sig[i] = size
        prev   = size

    out["signal"] = pd.Series(sig, index=out.index, dtype=float)

    # Returns with simple costs (no leverage)
    pos = pd.Series(out["signal"], index=out.index, dtype=float)
    turnover = pos.diff().abs().fillna(abs(pos))
    costs = turnover * cost_per_turn
    out["strat_ret"] = (pos.shift(1) * out["ret_cc"] - costs).fillna(0.0)

    return out

# ---------- Stats & plot ----------
def stats(r: pd.Series) -> dict:
    r = r.dropna()

    # ensure 1-D Series
    if isinstance(r, pd.DataFrame):
        if r.shape[1] == 0:
            return {"days": 0}
        r = r.iloc[:, 0]

    r = pd.to_numeric(r, errors="coerce").dropna()

    if len(r) == 0:
        return {"days": 0}

    curve = (1 + r).cumprod()
    ann_return = float(r.mean() * 252)
    ann_vol = float(r.std() * np.sqrt(252))
    sharpe = (ann_return / ann_vol) if ann_vol > 0 else np.nan
    max_dd = (1 - curve / curve.cummax()).max()

    return {
        "ann_return_pct": 100 * ann_return,
        "ann_vol_pct": 100 * ann_vol,
        "sharpe": float(sharpe),
        "total_return_pct": 100 * (curve.iloc[-1] - 1),
        "max_dd_pct": 100 * max_dd
    }



def backtest(years=5, start=None, outdir="outputs", make_plot=True):
    print("Downloading data...")
    data = fetch_data(start, years)
    print(f"Data range: {data.index.min().date()} → {data.index.max().date()} rows={len(data)}")
    df, features = build_features(data)

    print("Generating walk-forward probabilities...")
    train_days = 252  # ~1 trading year
    p_up = walk_forward_probs(df, features, initial_train_days=train_days)


    print("Applying strategy...")
    out = apply_strategy(df, p_up)

    # ---------- quick sanity checks ----------
    sig_corr = out["ret_cc"].corr(out["signal"].shift(1))
    # No leverage column in this version; positions are just the signal
    pos_corr = sig_corr

    long_mask  = (out["signal"].shift(1) > 0)
    ret_long   = out.loc[long_mask, "ret_cc"].mean()
    ret_not    = out.loc[~long_mask, "ret_cc"].mean()

    naive_curve = (1 + (out["signal"].shift(1) * out["ret_cc"]).fillna(0)).cumprod()
    naive_total = naive_curve.iloc[-1] - 1

    print("\n---- Alignment sanity checks ----")
    print(f"Corr(ret, signal-1d): {sig_corr:.3f}")
    print(f"Corr(ret, pos-1d)   : {pos_corr:.3f}")
    print(f"Avg SPY ret when long : {ret_long:.6f}  | when not-long : {ret_not:.6f}")
    print(f"Naive (sign-only, no costs/leverage) total: {naive_total*100:.2f}%")
    # ---------- end sanity checks ----------

    # Fresh adjusted SPY price aligned with df (for Buy & Hold)
    bh_px = yf.download(
    "SPY",
    start=str(df.index[0].date()),
    auto_adjust=True,
    progress=False
    )["Close"].reindex(df.index).ffill()


    print("\n=== Results ===")

    # --- Equity curves (full) ---
    eq_s_full = (1.0 + out["strat_ret"]).cumprod()
    bh_full   = bh_px / bh_px.iloc[0]

    # --- Start plot when the model first produces a probability (no double warm-up) ---
    first_live_date = p_up.dropna().index.min()
    start_plot_date = first_live_date

    # Optional: if you prefer to start when the strategy first takes a non-zero position,
    # use the first day with a non-zero signal at/after the live date.
    first_pos_idx = out.index[(out["signal"].fillna(0) != 0) & (out.index >= start_plot_date)]
    if len(first_pos_idx) > 0:
        start_plot_date = first_pos_idx[0]

    # (Optional) if you also want to require visible movement beyond a tiny epsilon,
    # uncomment next 3 lines to take the later of "train start" or "first move":
    # eps = 0.0005
    # moved = eq_s_full[(eq_s_full > 1 + eps) | (eq_s_full < 1 - eps)]
    # start_plot_date = max(start_plot_date, (moved.index.min() if len(moved) else start_plot_date))

    # --- Slice & rebase from that date ---
    eq_s_plot   = eq_s_full.loc[start_plot_date:]
    # REBASE strategy to 1.0 at the same start date so it aligns with SPY
    eq_s_plot   = eq_s_plot / eq_s_plot.iloc[0]

    bh_plot_px  = bh_px.loc[start_plot_date:]
    bh_plot     = bh_plot_px / bh_plot_px.iloc[0]

    # Compute stats from the same start date as the plot
    s_stats = stats(eq_s_plot.pct_change())
    bh_stats = stats(bh_plot.pct_change())

    print("Strategy :", s_stats)
    print("Buy&Hold :", bh_stats)
    print(f"Outperformance (total, pp): {(s_stats['total_return_pct'] - bh_stats['total_return_pct']):.2f}")

    # Create output folder
    os.makedirs(outdir, exist_ok=True)

    # Make a new numbered run folder inside outdir (test1, test2, ...)
    run_id = 1
    while True:
        run_folder = os.path.join(outdir, f"test{run_id}")
        if not os.path.exists(run_folder):
            break
        run_id += 1
    os.makedirs(run_folder)


    # --- Equity curves (plot) ---
    if make_plot:
        plt.figure(figsize=(10, 6))
        bh_plot.plot(label="SPY", alpha=0.75)
        eq_s_plot.plot(label="Strategy", alpha=0.95)
        plt.title("Equity Curves (Growth of $1)")
        plt.xlabel("Date"); plt.ylabel("Value"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(run_folder, "equity_curve.png"), dpi=150)
        plt.close()
        print(f"Saved {os.path.join(run_folder, 'equity_curve.png')}")

    # Save trades
    out.to_csv(os.path.join(run_folder, "trades.csv"))

    # Save summary
    with open(os.path.join(run_folder, "summary.json"), "w") as f:
        json.dump({
            "strategy_stats": s_stats,
            "buyhold_stats": bh_stats,
            "start_date": str(out.index.min().date()),
            "end_date": str(out.index.max().date())
        }, f, indent=2)
    print(f"Saved results in {run_folder}")

    # Save a copy of the current code into the run folder
    import shutil, sys
    code_path = sys.argv[0]  # path of the script being run
    code_copy = os.path.join(run_folder, os.path.basename(code_path) + ".txt")
    shutil.copy(code_path, code_copy)
    print(f"Saved code snapshot as text to {code_copy}")

    return s_stats, bh_stats, run_folder


# --------- Helper: summary for a given period ---------
def summarize_period(years=None, start=None, outdir="outputs", make_plot=False, label=""):
    s_stats, bh_stats, run_folder = backtest(
        years=years if years is not None else 5,
        start=start,
        outdir=outdir,
        make_plot=make_plot
    )
    tag = label or (f"{years}y" if years is not None else f"{start}→")
    print(f"\n=== Summary: {tag} ===")
    print("Strategy :", s_stats)
    print("Buy&Hold :", bh_stats)
    print(f"Outperformance (pp): {(s_stats['total_return_pct'] - bh_stats['total_return_pct']):.2f}")
    return s_stats, bh_stats, run_folder



# ---------- Windowed backtest helpers ----------
def backtest_for_window(start_date: str):
    """
    Run a self-contained backtest for a given start_date (string "YYYY-MM-DD").
    Returns: (title, eq_s_plot, bh_plot, s_stats, bh_stats)
    """
    data = fetch_data(start_date, years=None)
    df, features = build_features(data)
    train_days = 252
    p_up = walk_forward_probs(df, features, initial_train_days=train_days)
    out = apply_strategy(df, p_up)

    # Fresh adjusted SPY price aligned with df (for Buy & Hold)
    bh_px = yf.download(
        "SPY",
        start=str(df.index[0].date()),
        auto_adjust=True,
        progress=False
    )["Close"].reindex(df.index).ffill()

    # --- Equity curves (full) ---
    eq_s_full = (1.0 + out["strat_ret"]).cumprod()
    bh_full   = bh_px / bh_px.iloc[0]

    # --- Start plot when the model first produces a probability (no double warm-up) ---
    first_live_date = p_up.dropna().index.min()
    start_plot_date = first_live_date
    first_pos_idx = out.index[(out["signal"].fillna(0) != 0) & (out.index >= start_plot_date)]
    if len(first_pos_idx) > 0:
        start_plot_date = first_pos_idx[0]
    # --- Slice & rebase from that date ---
    eq_s_plot   = eq_s_full.loc[start_plot_date:]
    # REBASE strategy to 1.0 at the same start date so it aligns with SPY
    eq_s_plot   = eq_s_plot / eq_s_plot.iloc[0]

    bh_plot_px  = bh_px.loc[start_plot_date:]
    bh_plot     = bh_plot_px / bh_plot_px.iloc[0]
    # Compute stats from the same start date as the plot
    s_stats = stats(eq_s_plot.pct_change())
    bh_stats = stats(bh_plot.pct_change())
    # Title: "YYYY-MM-DD → YYYY-MM-DD"
    title = f"{start_date} → {df.index.max().date()}"
    return title, eq_s_plot, bh_plot, s_stats, bh_stats


# ---------- Multi-window panel backtest ----------
def multi_backtest(outdir="outputs_multi", make_plot=True):
    import csv
    windows = [
        ("2005-01-01", "Full 2005+"),
        ("2010-01-01", "Bullish 2010-2019+"),
        ("2015-01-01", "Mixed 2015+"),
        ("2020-01-01", "COVID era 2020+"),
        ("2022-01-01", "Bear 2022+"),
        ("2023-01-01", "Bull 2023+"),
    ]
    results = []
    eq_curves = []
    bh_curves = []
    labels = []
    sharpe_vals = []
    end_dates = []
    os.makedirs(outdir, exist_ok=True)
    for start, label in windows:
        print(f"Running window: {label} ({start})")
        title, eq_s_plot, bh_plot, s_stats, bh_stats = backtest_for_window(start)
        eq_curves.append(eq_s_plot)
        bh_curves.append(bh_plot)
        labels.append(label)
        sharpe_vals.append(s_stats.get('sharpe', float('nan')))
        end_dates.append(eq_s_plot.index[-1].date() if len(eq_s_plot) else "")
        results.append([
            label,
            start,
            str(eq_s_plot.index[-1].date()) if len(eq_s_plot) else "",
            f"{s_stats.get('sharpe', float('nan')):.2f}",
            f"{s_stats.get('total_return_pct', float('nan')):.2f}",
            f"{bh_stats.get('total_return_pct', float('nan')):.2f}",
            f"{s_stats.get('max_dd_pct', float('nan')):.2f}"
        ])
    # --- Plotting ---
    if make_plot:
        import math
        n = len(windows)
        ncols = 2
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.5 * nrows), sharey=False)
        axes = axes.flatten()
        for i, (eq, bh, label, sharpe) in enumerate(zip(eq_curves, bh_curves, labels, sharpe_vals)):
            ax = axes[i]
            bh.plot(ax=ax, label="SPY", alpha=0.75)
            eq.plot(ax=ax, label="Strategy", alpha=0.95)
            # Only show legend on first subplot
            if i == 0:
                ax.legend()
            ax.set_title(f"{label}  |  Sharpe {sharpe:.2f}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Value")
        # Hide unused axes
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "multi_equity.png"), dpi=150)
        plt.close()
    # --- Save summary CSV ---
    summary_path = os.path.join(outdir, "multi_summary.csv")
    with open(summary_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["label","start","end","strategy_sharpe","strategy_total_return_pct","buyhold_total_return_pct","strategy_max_dd_pct"])
        writer.writerows(results)
    if make_plot:
        print(f"Saved panel plot to {os.path.join(outdir, 'multi_equity.png')}")
    print(f"Saved summary to {summary_path}")









def live_signal(years=5, start=None):
    print("Preparing live signal...")
    data = fetch_data(start, years)
    df, features = build_features(data)
    p_up = walk_forward_probs(df, features)
    latest_date = df.index[-1]
    latest_p = float(p_up.iloc[-1])
    action = (
        "Go LONG at today’s close" if latest_p >= P_LONG_LIVE
        else ("Go SHORT at today’s close" if latest_p <= P_SHORT_LIVE else "Stay FLAT today")
    )
    print(json.dumps({
        "as_of": str(latest_date),
        "p_up": latest_p,
        "suggested_action": action,
        "notes": "Live view is informational only; no orders are placed."
    }, indent=2))

# ---------- CLI ----------
if __name__ == "__main__":
    # If no command-line arguments are supplied, run a full since-2005 backtest with plots.
    # This makes the script "click-to-run" friendly for GitHub users.
    if len(sys.argv) == 1:
        print("No arguments supplied. Running full backtest since 2005 with plots...")
        s_stats, bh_stats, run_folder = backtest(
            years=None, start="2005-01-01", outdir="outputs_full", make_plot=True
        )
        # Also create the multi-window equity panel in the same run folder
        try:
            multi_backtest(outdir=run_folder, make_plot=True)
        except Exception as e:
            print(f"[warn] Failed to create multi-window panel: {e}")
        raise SystemExit(0)

    ap = argparse.ArgumentParser(description="SPY timing bot — backtest and live signal")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--years", type=int, default=5, help="number of years back from today")
    group.add_argument("--start", type=str, help="YYYY-MM-DD fixed start date")
    ap.add_argument("--outdir", type=str, default="outputs", help="output folder")
    ap.add_argument("--live", action="store_true", help="print today’s signal instead of backtesting")
    ap.add_argument("--multi", action="store_true", help="run a panel of predefined windows and save a combined chart")
    ap.add_argument("--no-plot", action="store_true", help="skip saving charts (faster runs)")
    ap.add_argument("--compare", action="store_true", help="print a 5-year vs 2005+ comparison summary")
    args = ap.parse_args()

    make_plot = (not args.no_plot)

    if args.live:
        live_signal(years=args.years, start=args.start)
    elif args.compare:
        summarize_period(
            years=5, start=None,
            outdir=os.path.join(args.outdir, "cmp_5y"),
            make_plot=make_plot,
            label="Last 5y"
        )
        summarize_period(
            years=None, start="2005-01-01",
            outdir=os.path.join(args.outdir, "cmp_2005"),
            make_plot=make_plot,
            label="Since 2005"
        )
    elif args.multi:
        multi_backtest(outdir=args.outdir, make_plot=make_plot)
    else:
        backtest(years=args.years, start=args.start, outdir=args.outdir, make_plot=make_plot)
