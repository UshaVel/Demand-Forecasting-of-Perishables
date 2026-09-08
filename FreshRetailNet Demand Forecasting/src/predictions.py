# save predictions of all the model in a similar format #


import pandas as pd

import config as cf

#cols all the prediction file will have
COLUMNS=['store_id','product_id','dt','prediction','model_name',
         'input_version','split']


def save(pred_df,model_name,input_version,split):

    assert input_version in ('raw', 'recovered')
    assert split in ('val','test')

    out=pred_df.copy()

    #change date to proper format
    out['dt']=pd.to_datetime(out['dt']).dt.strftime('%Y-%m-%d')

    # a forecast below zero is set to zero
    out['prediction']=out['prediction'].clip(lower=0)

    # label the file 
    out['model_name'] = model_name
    out['input_version'] = input_version
    out['split'] = split

    out = out[COLUMNS]

    assert out['prediction'].notna().all(), 'some predictions are empty'
    assert len(out) == 350_000, f'expected 350,000 rows, got {len(out):,}'
    assert out.groupby(['store_id', 'product_id']).size().eq(cf.HORIZON).all(), \
        f'every series needs exactly {cf.HORIZON} days'

    if split == 'val':
        folder = cf.PRED_VAL
    else:
        folder = cf.PRED_TEST

    path = f'{folder}/{model_name}__{input_version}.parquet'
    out.to_parquet(path, index=False)
    print(f'[{split}] {len(out):,} rows : {model_name}__{input_version}')
    return path
