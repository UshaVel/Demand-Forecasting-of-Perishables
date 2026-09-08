# src/config.py
# ---------------------------------------------------------------------------
# Every setting for the forecasting stage lives here.
# Change something in this file and it changes everywhere. Nothing below this
# file should ever hard-code a path, a date, or a feature name.
#
# Usage in a notebook:
#     import sys; sys.path.append('/content/drive/MyDrive/Colab Notebooks/FreshRetailNet Demand Forecasting/src')
#     import config as cfg
#     cfg.check()          # confirms every input file is where it should be
# ---------------------------------------------------------------------------

import os

# === 1. THE ONLY LINE ANYONE ELSE NEEDS TO EDIT ============================
PROJECT = '/content/drive/MyDrive/Colab Notebooks/FreshRetailNet Demand Forecasting'

# === 2. FOLDERS (all derived from PROJECT) =================================
INPUTS      = f'{PROJECT}/inputs'
SRC         = f'{PROJECT}/src'
NOTEBOOKS   = f'{PROJECT}/notebooks'
PRED_VAL    = f'{PROJECT}/predictions/val'      # one file per model per input version
PRED_TEST   = f'{PROJECT}/predictions/test'
EVALUATION  = f'{PROJECT}/evaluation'           # results tables, environment record
RECOVERY    = f'{PROJECT}/Latent Demand Recovery'   # the frozen stage-1 code

# === 3. INPUT FILES ========================================================
RAW_TRAIN = f'{INPUTS}/freshretailnet_train.parquet'        # original data, 2024-03-28..06-25
RAW_TEST  = f'{INPUTS}/freshretailnet_test.parquet'         # held-out week, 2024-06-26..07-02
RECOVERED = f'{INPUTS}/recovered_demand_timesnet.parquet'   # RAW_TRAIN + 'sale_amount_pred' from TimesNet
MODEL_INPUT  = f'{INPUTS}/daily_demand.parquet'              # written by data.py; what the models read

# The recovery code runs as a SEPARATE PROCESS (!python app.py), so it cannot import
# this file. An environment variable is the one thing that crosses that boundary.
os.environ['FRN_TRAIN_PARQUET'] = RAW_TRAIN

# === 4. THE FORECASTING PROBLEM ===========================================
HORIZON = 7          # predict 7 days ahead

# Time splits. Models are fitted twice:
#   validation - train on data <= TUNE_TRAIN_END, score on VAL_START..VAL_END
#   test       - train on everything through VAL_END, score on the held-out week
# Validation is where every choice is made (model selection, routing, hyperparameters).
# The test week is scored ONCE, at the end.
# Every date the project uses is written here, in order.
TRAIN_START    = '2024-03-28'   # first day of history
TUNE_TRAIN_END = '2024-06-18'   # a validation fit trains up to here
VAL_START      = '2024-06-19'
VAL_END        = '2024-06-25'   # a test fit trains up to here
TEST_START     = '2024-06-26'
TEST_END       = '2024-07-02'   # last day in the data

# === 5. FEATURES (identical for every model — this is what keeps it fair) ==
# Known for the future: promotions come from marketing plans, weather from forecasts,
# holidays from the calendar. The paper makes the same assumption.
EXOG = ['discount', 'holiday_flag', 'activity_flag', 'precpt',
        'avg_temperature', 'avg_humidity', 'avg_wind_level']

# Fixed per series. store_id and product_id are deliberately EXCLUDED: neuralforecast
# feeds statics in as numbers, so high-cardinality IDs would imply a false ordering
# (product 700 > product 300) without giving the model real embeddings.
STATIC = ['city_id', 'management_group_id', 'first_category_id',
          'second_category_id', 'third_category_id']

# === 6. SHARED MODEL SETTINGS =============================================
INPUT_SIZE = 60          # days of history each model sees
MAX_STEPS  = 500         # training length for neuralforecast models
SCALER     = 'standard'  # NOT 'robust' (the library default): robust scaling measurably
                         # hurt on this data, most on low-volume series. See the ablation.
SEED       = 2025

# === 7. HELPERS ============================================================
def ensure_dirs():
    """Create the output folders if they do not exist yet."""
    for d in (INPUTS, PRED_VAL, PRED_TEST, EVALUATION):
        os.makedirs(d, exist_ok=True)

def check():
    """Confirm every input file is where this config says it is.
    Run at the top of each notebook — it catches a wrong path in one second
    instead of three hours into a training run."""
    ensure_dirs()
    ok = True
    for name in ['RAW_TRAIN', 'RAW_TEST', 'RECOVERED']:
        path = globals()[name]
        exists = os.path.exists(path)
        size = f'{os.path.getsize(path)/1e6:8.1f} MB' if exists else '  MISSING'
        print(f'  {name:10s} {size}  {path}')
        ok &= exists
    print('\nconfig OK' if ok else '\n*** SOME INPUTS ARE MISSING — fix the paths above before continuing ***')
    return ok