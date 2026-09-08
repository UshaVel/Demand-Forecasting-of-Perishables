

import numpy as np
import pandas as pd

import config as cf

# ======= Array columns ===== #

array_cols=['hours_sale','hours_stock_status']

# == BUILD TWO MODEL INPUTS == #
def build(verbose=True):

    train=pd.read_parquet(cf.RECOVERED) # 4,500,000 rows (Mar 28 to Jun 25)
    test=pd.read_parquet(cf.RAW_TEST)   # 350k rows (Jun 26 to Jul 02)

    # Timesnet recovery ran on train data only
    # so adding it as nan in test

    test['sale_amount_pred']=np.nan

    # union of train and test
    df=pd.concat([train,test],ignore_index=True)
    del train , test

    # drop the array columns
    df=df.drop(columns=array_cols)

    # fields needed for neuralforecast

    df['ds']=pd.to_datetime(df['dt'])
    df['unique_id']=df['store_id'].astype(str)+'_'+ df['product_id'].astype(str)

    # data types
    number_cols=cf.EXOG + ['sale_amount','sale_amount_pred','stock_hour6_22_cnt']
    for col in number_cols:
        df[col]=df[col].astype('float32')
    for col in cf.STATIC:
        df[col]=df[col].astype('int32')

    # one series at a time, in date order
    df=df.sort_values(['unique_id','ds']).reset_index(drop=True)

    check(df,verbose=verbose)
    df.to_parquet(cf.MODEL_INPUT,index=False)
    if verbose:
        print('\nsaved:' ,cf.MODEL_INPUT)

    return df


# === CHECK === #

def check(df,verbose=True):

    assert len(df)==4_850_000, f'Expected 4,850,000 rows, got {len(df):,}'
    assert df['unique_id'].nunique()==50_000, 'Expected 50,000 series'
    assert df['ds'].nunique()==97,'Expected 97 days (90 train and 7 test)'

    assert str(df['ds'].min().date())==cf.TRAIN_START, 'Different start date than train start'
    assert str(df['ds'].max().date())==cf.TEST_END, 'Different end date than test end'

    assert not df.duplicated(['unique_id','ds']).any(), 'date is duplicated'
    assert df.groupby('unique_id').size().eq(97).all(),'a series is missing some dates'

    # checking if raw target is present

    assert df['sale_amount'].notna().all(), 'sale_amount has empty values'

    # recovered target should be empty on test week only

    test_week=df['ds']>cf.VAL_END
    assert (df['sale_amount_pred'].isna()==test_week).all(), 'target is empty outside test'

    # checking if covariates has empty data
    for col in cf.EXOG + cf.STATIC:
        assert df[col].notna().all(), f'{col} has empty values'

    if verbose:
        train = df[(df['ds'] >= cf.TRAIN_START) & (df['ds'] <= cf.TUNE_TRAIN_END)]
        val = df[(df['ds'] >= cf.VAL_START) & (df['ds'] <= cf.VAL_END)]
        test = df[(df['ds'] >= cf.TEST_START) & (df['ds'] <= cf.TEST_END)]
# to get mean of train including val period
        history = df[df['ds'] <= cf.VAL_END]

        print('rows   ', f'{len(df):,}')
        print('series ', f'{df["unique_id"].nunique():,}')
        print('dates  ', df['ds'].min().date(), 'to', df['ds'].max().date())
        print('train  ', f'{len(train):,}', train['ds'].min().date(), 'to', train['ds'].max().date())
        print('val    ', f'{len(val):,}', val['ds'].min().date(), 'to', val['ds'].max().date())
        print('test   ', f'{len(test):,}', test['ds'].min().date(), 'to', test['ds'].max().date())
        print('Sale amount mean-raw', round(history['sale_amount'].mean(), 4))
        print('Sale amount mean-recovered', round(history['sale_amount_pred'].mean(), 4), '(on train data only)')


# ==== LOAD ==== #

def load():
    # read the model input table
    return pd.read_parquet(cf.MODEL_INPUT)


def splittbl(df, input_version, split):

    """input_version : 'raw'       -> y = sale_amount
                    'recovered' -> y = sale_amount_pred
    split         : 'val'       -> train up to Jun 18, predict Jun 19-25
                    'test'      -> train up to Jun 25, predict Jun 26 - Jul 02"""

    assert input_version in ('raw','recovered')
    assert split in ('val','test')
# get daatset for raw or recovered based on input version
    if input_version=='raw':
        target='sale_amount'
    else:
        target='sale_amount_pred'
# splitting the dataset for test and validation run
    if split == 'val':
        train_start,train_end=cf.TRAIN_START,cf.TUNE_TRAIN_END
        predict_start,predict_end=cf.VAL_START,cf.VAL_END
    else:
        train_start,train_end=cf.TRAIN_START,cf.VAL_END
        predict_start,predict_end=cf.TEST_START,cf.TEST_END
#copyig the tables
    is_train=(df['ds'] >= train_start) & (df['ds'] <= train_end)
    train_df = df[is_train].copy()
    train_df['y'] = train_df[target]
    train_df = train_df[['unique_id', 'ds', 'y'] + cf.EXOG + cf.STATIC]
    assert train_df['y'].notna().all(), f'{target} is empty inside the training window'

    is_predicted = (df['ds'] >= predict_start) & (df['ds'] <= predict_end)
    predict_df = df[is_predicted][['unique_id', 'ds'] + cf.EXOG].copy()
    assert predict_df.groupby('unique_id').size().eq(cf.HORIZON).all(), \
        f'the prediction table is not exactly {cf.HORIZON} days per series'

    return train_df, predict_df


if __name__ == '__main__':
    build()