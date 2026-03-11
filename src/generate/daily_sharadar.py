"""
Daily Sharadar data loading for IPCA / GIPCA experiments.

Universe : S&P 500 constituents (point-in-time membership)
Returns  : SHARADAR_SEP  – total return = (close + div) / prev_close - 1
Chars    : 20 firm characteristics from SEP, DAILY, SF1
Macro    : Fama-French 5 daily factors
"""

import bisect
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata

# ── 20 characteristics in panel order ───────────────────────────────
CHAR_NAMES = [
    'mvel1',    # Size:         marketcap                     (DAILY)
    'bm',       # Value:        1 / pb                        (DAILY)
    'ep',       # Value:        1 / pe                        (DAILY)
    'sp',       # Value:        1 / ps                        (DAILY)
    'roaq',     # Profitability: netinccmn / assets            (SF1)
    'roeq',     # Profitability: netinccmn / equity            (SF1)
    'gma',      # Profitability: gp / revenue                  (SF1)
    'operprof', # Profitability: opinc / assets                (SF1)
    'agr',      # Investment:   YoY asset growth               (SF1)
    'sgr',      # Growth:       YoY revenue growth             (SF1)
    'lev',      # Leverage:     debt / equity                  (SF1)
    'currat',   # Liquidity:    currentratio                   (SF1)
    'chcsho',   # Investment:   YoY shares change              (SF1)
    'cfp',      # Value:        (netinccmn + depamor) / mktcap (SF1+DAILY)
    'depr',     # Balance:      depamor / assets               (SF1)
    'dy',       # Value:        divyield                       (SF1)
    'mom1m',    # Momentum:     21-day cumulative return       (SEP)
    'mom6m',    # Momentum:     126-day cumulative return      (SEP)
    'retvol',   # Risk:         21-day return std              (SEP)
    'turn',     # Liquidity:    21-day avg vol / sharesbas     (SEP)
]

MACRO_NAMES = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']


# ════════════════════════════════════════════════════════════════════
# S&P 500 Membership
# ════════════════════════════════════════════════════════════════════

def load_sp500_membership(sp500_path):
    """
    Build point-in-time S&P 500 membership from SHARADAR_SP500.csv.

    Uses quarterly ``historical`` / ``current`` snapshots as anchors and
    applies inter-snapshot ``added`` / ``removed`` events.

    Returns
    -------
    timeline_dates : list[pd.Timestamp]
        Sorted dates at which membership changes.
    timeline_sets : list[frozenset[str]]
        Corresponding member-ticker sets.
    all_tickers : set[str]
        Union of all tickers ever in S&P 500.
    """
    df = pd.read_csv(sp500_path, parse_dates=['date'])

    # Quarterly snapshots – authoritative full lists
    snap_df = df[df['action'].isin(['historical', 'current'])]
    snap_dates = sorted(snap_df['date'].unique())

    # Add / remove events
    events = df[df['action'].isin(['added', 'removed'])].sort_values('date')

    timeline = {}

    for i, snap_date in enumerate(snap_dates):
        current = set(snap_df.loc[snap_df['date'] == snap_date, 'ticker'])
        timeline[snap_date] = current.copy()

        next_snap = (snap_dates[i + 1] if i < len(snap_dates) - 1
                     else pd.Timestamp('2099-12-31'))
        mask = (events['date'] > snap_date) & (events['date'] < next_snap)

        for event_date in sorted(events.loc[mask, 'date'].unique()):
            day_ev = events.loc[mask & (events['date'] == event_date)]
            for _, ev in day_ev.iterrows():
                if ev['action'] == 'added':
                    current.add(ev['ticker'])
                else:
                    current.discard(ev['ticker'])
            timeline[event_date] = current.copy()

    sorted_items = sorted(timeline.items())
    timeline_dates = [d for d, _ in sorted_items]
    timeline_sets = [frozenset(s) for _, s in sorted_items]
    all_tickers = set().union(*timeline_sets)

    return timeline_dates, timeline_sets, all_tickers


def _membership_on_date(timeline_dates, timeline_sets, target_date):
    """Binary-search for the S&P 500 member set on *target_date*."""
    idx = bisect.bisect_right(timeline_dates, target_date) - 1
    return timeline_sets[idx] if idx >= 0 else frozenset()


# ════════════════════════════════════════════════════════════════════
# Individual Data Loaders
# ════════════════════════════════════════════════════════════════════

def load_daily_returns(sep_path, sp500_tickers, start_date, end_date):
    """
    Load daily stock prices from SHARADAR_SEP.csv for *sp500_tickers*.

    Total return: ``ret = (close + dividends) / prev_close - 1``

    Extra 8-month look-back is loaded for rolling-window characteristics.

    Returns
    -------
    DataFrame  (ticker, date, ret, close, volume)
    """
    sp500_set = set(sp500_tickers)
    read_start = pd.Timestamp(start_date) - pd.DateOffset(months=8)
    end_dt = pd.Timestamp(end_date)

    parts = []
    for chunk in pd.read_csv(sep_path,
                             usecols=['ticker', 'date', 'close',
                                      'volume', 'dividends'],
                             parse_dates=['date'],
                             chunksize=1_000_000):
        filt = chunk[(chunk['ticker'].isin(sp500_set)) &
                     (chunk['date'] >= read_start) &
                     (chunk['date'] <= end_dt)]
        if len(filt):
            parts.append(filt)

    df = pd.concat(parts, ignore_index=True)
    df.sort_values(['ticker', 'date'], inplace=True)

    df['prev_close'] = df.groupby('ticker')['close'].shift(1)
    df['ret'] = (df['close'] + df['dividends']) / df['prev_close'] - 1
    df.dropna(subset=['ret'], inplace=True)

    return df[['ticker', 'date', 'ret', 'close', 'volume']].reset_index(drop=True)


def load_daily_valuations(daily_path, sp500_tickers, start_date, end_date):
    """
    Load daily valuation ratios from SHARADAR_DAILY.csv.

    Computes ``mvel1 = marketcap``, ``bm = 1/pb``, ``ep = 1/pe``, ``sp = 1/ps``.

    Returns
    -------
    DataFrame  (ticker, date, marketcap, bm, ep, sp)
    """
    sp500_set = set(sp500_tickers)
    read_start = pd.Timestamp(start_date) - pd.DateOffset(months=8)
    end_dt = pd.Timestamp(end_date)

    parts = []
    for chunk in pd.read_csv(daily_path,
                             usecols=['ticker', 'date', 'marketcap',
                                      'pb', 'pe', 'ps'],
                             parse_dates=['date'],
                             chunksize=1_000_000):
        filt = chunk[(chunk['ticker'].isin(sp500_set)) &
                     (chunk['date'] >= read_start) &
                     (chunk['date'] <= end_dt)]
        if len(filt):
            parts.append(filt)

    df = pd.concat(parts, ignore_index=True)
    df['bm'] = 1.0 / df['pb'].replace(0, np.nan)
    df['ep'] = 1.0 / df['pe'].replace(0, np.nan)
    df['sp'] = 1.0 / df['ps'].replace(0, np.nan)

    return df[['ticker', 'date', 'marketcap', 'bm', 'ep', 'sp']].reset_index(drop=True)


def load_quarterly_fundamentals(sf1_path, sp500_tickers):
    """
    Load quarterly fundamentals from SHARADAR_SF1.csv (``dimension='ARQ'``).

    Uses ``datekey`` (SEC filing date) for point-in-time correctness.

    Level chars : roaq, roeq, gma, operprof, lev, currat, depr, dy, cfp_num
    Growth chars: agr (assets), sgr (revenue), chcsho (shares) — YoY

    Returns
    -------
    DataFrame  (ticker, datekey, <chars>, sharesbas)
    """
    needed_cols = [
        'ticker', 'dimension', 'calendardate', 'datekey',
        'netinccmn', 'assets', 'equity', 'gp', 'revenue',
        'opinc', 'debt', 'currentratio', 'sharesbas',
        'depamor', 'divyield',
    ]
    sp500_set = set(sp500_tickers)

    parts = []
    for chunk in pd.read_csv(sf1_path, usecols=needed_cols,
                             chunksize=1_000_000):
        filt = chunk[(chunk['dimension'] == 'ARQ') &
                     (chunk['ticker'].isin(sp500_set))]
        if len(filt):
            parts.append(filt)

    df = pd.concat(parts, ignore_index=True)
    df['datekey'] = pd.to_datetime(df['datekey'])
    df['calendardate'] = pd.to_datetime(df['calendardate'])
    df.sort_values(['ticker', 'calendardate'], inplace=True)

    # Level characteristics
    _safe = lambda s: s.replace(0, np.nan)
    df['roaq']     = df['netinccmn'] / _safe(df['assets'])
    df['roeq']     = df['netinccmn'] / _safe(df['equity'])
    df['gma']      = df['gp']        / _safe(df['revenue'])
    df['operprof'] = df['opinc']     / _safe(df['assets'])
    df['lev']      = df['debt']      / _safe(df['equity'])
    df['currat']   = df['currentratio']
    df['depr']     = df['depamor']   / _safe(df['assets'])
    df['dy']       = df['divyield']
    df['cfp_num']  = df['netinccmn'].fillna(0) + df['depamor'].fillna(0)

    # Growth characteristics — YoY (shift 4 quarters)
    for col, src in [('agr', 'assets'), ('sgr', 'revenue'),
                     ('chcsho', 'sharesbas')]:
        lagged = df.groupby('ticker')[src].shift(4)
        df[col] = (df[src] - lagged) / lagged.abs().replace(0, np.nan)

    out_cols = ['ticker', 'datekey',
                'roaq', 'roeq', 'gma', 'operprof', 'lev', 'currat',
                'depr', 'dy', 'cfp_num', 'agr', 'sgr', 'chcsho',
                'sharesbas']
    return df[out_cols].reset_index(drop=True)


def load_ff5_daily(ff5_path, start_date, end_date):
    """
    Load Fama-French 5 daily factors (values returned in **decimal**).

    Returns
    -------
    macro : DataFrame  (index=Date, cols=[Mkt-RF, SMB, HML, RMW, CMA])
    rf    : Series     (index=Date)
    """
    df = pd.read_csv(ff5_path)
    df.columns = df.columns.str.strip()
    df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'].astype(str).str.strip(),
                                format='%Y%m%d')
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)

    return df[MACRO_NAMES] / 100.0, df['RF'] / 100.0


# ════════════════════════════════════════════════════════════════════
# Panel Construction
# ════════════════════════════════════════════════════════════════════

def build_daily_panel(
    data_dir,
    start_date='2015-01-01',
    end_date='2018-12-31',
    verbose=True,
):
    """
    Build a balanced daily panel for S&P 500 stocks.

    Parameters
    ----------
    data_dir : str or Path
        Root data directory (parent of ``daily/sharadar_db/``).
    start_date, end_date : str
        Panel date range.
    verbose : bool
        Print progress messages.

    Returns
    -------
    dict
        returns_arr : ndarray (T, N)  – daily excess returns
        chars_arr   : ndarray (T, N, 20) – rank-transformed characteristics
        macro_arr   : ndarray (T, 5)  – FF5 factors (decimal)
        dates       : ndarray of datetime64
        tickers     : ndarray of str
        char_names  : list[str]
        macro_names : list[str]
        rf          : ndarray (T,) – daily risk-free rate (decimal)
    """
    data_dir = Path(data_dir)
    sdir = data_dir / 'daily' / 'sharadar_db'
    start_dt = pd.Timestamp(start_date)

    # ── 1. S&P 500 membership ──────────────────────────────────────
    if verbose:
        print("1/10  Loading S&P 500 membership …")
    tl_dates, tl_sets, all_tickers = load_sp500_membership(
        sdir / 'SHARADAR_SP500.csv')
    if verbose:
        print(f"       {len(all_tickers)} unique tickers ever in S&P 500")

    # ── 2. Daily returns ───────────────────────────────────────────
    if verbose:
        print("2/10  Loading daily returns (SEP) …")
    ret_df = load_daily_returns(sdir / 'SHARADAR_SEP.csv',
                                all_tickers, start_date, end_date)
    if verbose:
        print(f"       {len(ret_df):,} return observations loaded")

    # ── 3. Point-in-time S&P 500 filter ────────────────────────────
    if verbose:
        print("3/10  Filtering to S&P 500 members …")
    member_cache = {
        d: _membership_on_date(tl_dates, tl_sets, d)
        for d in ret_df['date'].unique()
    }
    mask = np.array([
        t in member_cache[d]
        for t, d in zip(ret_df['ticker'].values, ret_df['date'].values)
    ])
    ret_df = ret_df[mask].copy()
    if verbose:
        print(f"       {len(ret_df):,} after S&P 500 filter")

    # ── 4. Daily valuations ────────────────────────────────────────
    if verbose:
        print("4/10  Loading daily valuations (DAILY) …")
    val_df = load_daily_valuations(sdir / 'SHARADAR_DAILY.csv',
                                   all_tickers, start_date, end_date)
    if verbose:
        print(f"       {len(val_df):,} valuation observations")

    # ── 5. Quarterly fundamentals ──────────────────────────────────
    if verbose:
        print("5/10  Loading quarterly fundamentals (SF1) …")
    fund_df = load_quarterly_fundamentals(sdir / 'SHARADAR_SF1.csv',
                                          all_tickers)
    if verbose:
        print(f"       {len(fund_df):,} quarterly observations")

    # ── 6. Merge returns + valuations ──────────────────────────────
    if verbose:
        print("6/10  Merging data sources …")
    panel = ret_df.merge(val_df, on=['ticker', 'date'], how='left')

    # ── 7. Forward-fill quarterly fundamentals (merge_asof) ────────
    if verbose:
        print("7/10  Forward-filling quarterly fundamentals …")
    # merge_asof requires the 'on' key to be globally sorted
    panel.sort_values('date', inplace=True)
    fund_renamed = fund_df.rename(columns={'datekey': 'date'}).sort_values('date')

    panel = pd.merge_asof(
        panel,
        fund_renamed,
        on='date', by='ticker', direction='backward',
    )

    # cfp = (netinccmn + depamor) / marketcap
    panel['cfp'] = panel['cfp_num'] / panel['marketcap'].replace(0, np.nan)

    # ── 8. Rolling characteristics from SEP ────────────────────────
    if verbose:
        print("8/10  Computing rolling characteristics …")
    panel.sort_values(['ticker', 'date'], inplace=True)

    # Momentum via rolling sum of log-returns
    panel['_log_ret'] = np.log1p(panel['ret'])
    panel['mom1m'] = np.expm1(
        panel.groupby('ticker')['_log_ret']
             .transform(lambda x: x.rolling(21, min_periods=15).sum())
    )
    panel['mom6m'] = np.expm1(
        panel.groupby('ticker')['_log_ret']
             .transform(lambda x: x.rolling(126, min_periods=90).sum())
    )

    # Volatility
    panel['retvol'] = (
        panel.groupby('ticker')['ret']
             .transform(lambda x: x.rolling(21, min_periods=15).std())
    )

    # Turnover = 21-day avg volume / shares outstanding
    panel['_avg_vol'] = (
        panel.groupby('ticker')['volume']
             .transform(lambda x: x.rolling(21, min_periods=15).mean())
    )
    panel['turn'] = panel['_avg_vol'] / panel['sharesbas'].replace(0, np.nan)

    # mvel1 alias
    panel['mvel1'] = panel['marketcap']

    # ── Lag all characteristics by 1 day ───────────────────────────
    # Ensures chars[t] use only information available at end of day
    # t-1, avoiding contemporaneous bias (e.g. mom1m[t] ∋ ret[t]).
    lag_cols = list(CHAR_NAMES)
    panel[lag_cols] = panel.groupby('ticker')[lag_cols].shift(1)

    # ── Trim look-back period ──────────────────────────────────────
    panel = panel[panel['date'] >= start_dt].copy()

    # ── 9. Pivot to balanced arrays ────────────────────────────────
    if verbose:
        print("9/10  Pivoting to balanced arrays …")

    panel.drop_duplicates(subset=['ticker', 'date'], keep='last', inplace=True)

    dates = sorted(panel['date'].unique())
    tickers = sorted(panel['ticker'].unique())
    T, N = len(dates), len(tickers)
    L = len(CHAR_NAMES)

    ret_pivot = panel.pivot(index='date', columns='ticker', values='ret')
    ret_pivot = ret_pivot.reindex(index=dates, columns=tickers)
    returns_arr = ret_pivot.values.astype(np.float64)

    chars_arr = np.full((T, N, L), np.nan, dtype=np.float64)
    for l, cname in enumerate(CHAR_NAMES):
        cpiv = panel.pivot(index='date', columns='ticker', values=cname)
        cpiv = cpiv.reindex(index=dates, columns=tickers)
        chars_arr[:, :, l] = cpiv.values

    # ── 10. Cross-sectional rank transform ─────────────────────────
    if verbose:
        print("10/10 Rank-transforming characteristics …")
    for t in range(T):
        for l in range(L):
            col = chars_arr[t, :, l]
            valid = ~np.isnan(col)
            n_valid = valid.sum()
            if n_valid > 1:
                chars_arr[t, valid, l] = rankdata(col[valid]) / n_valid - 0.5
    chars_arr = np.nan_to_num(chars_arr, nan=0.0)

    # ── FF5 daily factors + excess returns ─────────────────────────
    macro_df, rf_series = load_ff5_daily(
        data_dir / 'F-F_Research_Data_5_Factors_2x3_daily.csv',
        start_date, end_date,
    )

    dates_idx = pd.DatetimeIndex(dates)
    common_dates = dates_idx.intersection(macro_df.index)
    date_mask = dates_idx.isin(common_dates)

    returns_arr = returns_arr[date_mask]
    chars_arr = chars_arr[date_mask]
    macro_arr = macro_df.reindex(common_dates).values.astype(np.float64)
    rf_arr = rf_series.reindex(common_dates).values.astype(np.float64)

    # Excess returns = ret − RF
    nan_ret = np.isnan(returns_arr)
    returns_arr = returns_arr - rf_arr[:, None]
    returns_arr = np.nan_to_num(returns_arr, nan=0.0)

    T_final = returns_arr.shape[0]

    if verbose:
        pct_nonzero = (returns_arr != 0).mean()
        print(f"\nPanel summary:")
        print(f"  T = {T_final} trading days")
        print(f"  N = {N} stocks")
        print(f"  L = {L} characteristics")
        print(f"  R = {len(MACRO_NAMES)} macro factors")
        print(f"  Dates: {common_dates[0].date()} — {common_dates[-1].date()}")
        print(f"  Non-zero returns: {pct_nonzero:.1%}")
        print(f"  Characteristic coverage (non-zero %):")
        for l, cn in enumerate(CHAR_NAMES):
            nz = (chars_arr[:, :, l] != 0).mean()
            print(f"    {cn:>10s}: {nz:.1%}")

    return {
        'returns_arr': returns_arr,    # (T, N)
        'chars_arr':   chars_arr,      # (T, N, 20)
        'macro_arr':   macro_arr,      # (T, 5)
        'dates':       common_dates.values,
        'tickers':     np.array(tickers),
        'char_names':  list(CHAR_NAMES),
        'macro_names': list(MACRO_NAMES),
        'rf':          rf_arr,         # (T,)
    }
