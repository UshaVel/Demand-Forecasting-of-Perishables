# src/evaluate.py
# ---------------------------------------------------------------------------
# Turns the prediction files into a results table.
#
# Two rules that decide what a score means, both from the paper:
#   1. the answer is always the OBSERVED sale_amount - never the recovered
#      column, for either branch
#   2. only days with no stockout count. On a stockout day the recorded sales
#      are not the real demand, so scoring against them is meaningless.
# ---------------------------------------------------------------------------

import os
import pandas as pd

import numpy as np
import config as cfg
import data


def truth(split):
    """The answer sheet for one window: observed sales on non-stockout days."""
    df = data.load()

    if split == 'val':
        start, end = cfg.VAL_START, cfg.VAL_END
    else:
        start, end = cfg.TEST_START, cfg.TEST_END

    g = df[(df['ds'] >= start) & (df['ds'] <= end)].copy()
    g = g[g['stock_hour6_22_cnt'] == 0]              # no stockout that day
    g['dt'] = g['ds'].dt.strftime('%Y-%m-%d')        # match the prediction files
    return g[['store_id', 'product_id', 'dt', 'sale_amount']]


def load_predictions(split):
    """Read every prediction file in one folder into a single table."""
    folder = cfg.PRED_VAL if split == 'val' else cfg.PRED_TEST
    frames = []

    for f in sorted(os.listdir(folder)):
        if not f.endswith('.parquet'):
            continue
        # 'LightGBM__raw.parquet' -> model 'LightGBM', input 'raw'
        model, input_version = f[:-8].rsplit('__', 1)
        p = pd.read_parquet(f'{folder}/{f}')[['store_id', 'product_id', 'dt', 'prediction']]
        p['model'] = model
        p['input'] = input_version
        frames.append(p)

    return pd.concat(frames, ignore_index=True)


# pooled over all rows - NOT the average of per-series scores
def wape(g):
    return (g['prediction'] - g['sale_amount']).abs().sum() / g['sale_amount'].sum()

def wpe(g):
    """Same, but signed: negative = under-forecasting, positive = over."""
    return (g['prediction'] - g['sale_amount']).sum() / g['sale_amount'].sum()


def results(split):
    """One row per model per input version, plus the raw-vs-recovered table."""
    t = truth(split)
    p = load_predictions(split)
    m = p.merge(t, on=['store_id', 'product_id', 'dt'], how='inner')

    rows = []
    for (model, input_version), g in m.groupby(['model', 'input']):
        rows.append({'model': model, 'input': input_version,
                     'WAPE': round(wape(g), 4), 'WPE': round(wpe(g), 4),
                     'rows': len(g)})
    long = pd.DataFrame(rows)

    wide = long.pivot(index='model', columns='input', values='WAPE')
    wide['gain'] = wide['raw'] - wide['recovered']     # + means recovery helped
    return long, wide.sort_values('recovered')


def scale_factors(df, split):
    """Seasonal-naive error on the training history, one value per series.
    The 'repeat last week' benchmark that MASE and RMSSE divide by.

    The window stops at the forecast origin for the split being scored, so the
    week under evaluation is never inside its own scale factor. This matches
    the expanding-window protocol used everywhere else:
        val  -> 28 Mar .. 18 Jun
        test -> 28 Mar .. 25 Jun"""
    end = cfg.TUNE_TRAIN_END if split == 'val' else cfg.VAL_END
    h = df[df['ds'] <= end].sort_values(['unique_id', 'ds'])
    d = h.groupby('unique_id')['sale_amount'].diff(7)
    mae_s = d.abs().groupby(h['unique_id']).mean()
    mse_s = (d ** 2).groupby(h['unique_id']).mean()
    return mae_s[mae_s > 0], mse_s[mse_s > 0]


def extra_metrics(split):
    """MASE, RMSSE and R2 for every model. Rescores the SAME saved files
    that results() uses - nothing is re-run, WAPE is untouched."""
    t = truth(split)
    p = load_predictions(split)
    m = p.merge(t, on=['store_id', 'product_id', 'dt'], how='inner')
    m['unique_id'] = m['store_id'].astype(str) + '_' + m['product_id'].astype(str)

    mae_s, mse_s = scale_factors(data.load(),split)

    rows = []
    for (model, iv), g in m.groupby(['model', 'input']):
        err = g['prediction'] - g['sale_amount']

        # per-series first, then the MEDIAN across series
        # (mean is unstable on low-volume series)
        per_abs = err.abs().groupby(g['unique_id']).mean()
        per_sq = (err ** 2).groupby(g['unique_id']).mean()
        mase = (per_abs / mae_s.reindex(per_abs.index)).dropna().median()
        rmsse = np.sqrt((per_sq / mse_s.reindex(per_sq.index)).dropna()).median()

        ss_res = (err ** 2).sum()
        ss_tot = ((g['sale_amount'] - g['sale_amount'].mean()) ** 2).sum()

        rows.append({'model': model, 'input': iv,
                     'MASE': round(mase, 4),
                     'RMSSE': round(rmsse, 4),
                     'R2': round(1 - ss_res / ss_tot, 4)})

    return pd.DataFrame(rows)