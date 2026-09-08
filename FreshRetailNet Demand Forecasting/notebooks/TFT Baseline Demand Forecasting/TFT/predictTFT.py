import os, numpy as np, pandas as pd, torch, argparse
import configs.tft_config as config
from dataset.dataset import Dataset
from models.tft.model import TemporalFusionTransformer
from datetime import datetime

_orig = torch.load
torch.load = lambda *a, **k: _orig(*a, **{**k, 'weights_only': False})

TRAIN = os.environ.get("FRN_TRAIN_PARQUET",
    "/content/drive/MyDrive/Colab Notebooks/FreshRetailNet Demand Forecasting/inputs/freshretailnet_train.parquet")
TEST = os.environ.get("FRN_TEST_PARQUET",
    "/content/drive/MyDrive/Colab Notebooks/FreshRetailNet Demand Forecasting/inputs/freshretailnet_test.parquet")
PRED_DIR = os.environ.get("FRN_PRED_DIR",
    "/content/drive/MyDrive/Colab Notebooks/FreshRetailNet Demand Forecasting/predictions/test")

os.makedirs(PRED_DIR, exist_ok=True)
config.date='2024-06-26'; config.quantiles=7; config.use_gpu=True; config.num_workers=0   # 0 = safest
config.dataset_config["max_prediction_length"]=7; config.dataset_config["max_encoder_length"]=70
dates=[d.strftime("%Y-%m-%d") for d in pd.date_range(config.date, periods=7)]

def loadDataset(data_type, path):
    print('1) loading context parquet...', flush=True)
    if data_type=='recovered':
        df=pd.read_parquet(path); df['sale_amount']=df['sale_amount_pred']
    else:
        df=pd.read_parquet(TRAIN)
    print('   context:', df.shape, flush=True)
    print('2) loading test parquet...', flush=True)
    ev=pd.read_parquet(TEST); print('   test:', ev.shape, flush=True)
    print('3) concat...', flush=True)
    df=pd.concat([df,ev],ignore_index=True); print('   concat:', df.shape, flush=True)
    df['day_of_week']=df['dt'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d').weekday())
    df.loc[df.dt>=config.date,'sale_amount']=np.nan
    print('4) building TimeSeriesDataSet (this is the heavy step)...', flush=True)
    ds=Dataset(df,config); print('   dataset built OK', flush=True)
    return ds

if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument("--demand_path", default=os.environ.get(
        "FRN_DEMAND_PARQUET",
        "/content/drive/MyDrive/Colab Notebooks/FreshRetailNet Demand Forecasting/Latent Demand Recovery/exp/demand/demand.parquet"))
    ap.add_argument("--demand", action='store_true')
    a=ap.parse_args(); dt='recovered' if a.demand else 'censored'; iv='recovered' if a.demand else 'raw'
    ds=loadDataset(dt, a.demand_path)
    ck=sorted(os.listdir(f"./lightning_logs/{dt}/checkpoints"))[-1]; print('5) checkpoint:', ck, flush=True)
    m=TemporalFusionTransformer.load_from_checkpoint(f"./lightning_logs/{dt}/checkpoints/{ck}").cuda()
    print('6) running predict...', flush=True)
    pr,ix=m.predict(ds.predict_df,return_index=True); pr=np.asarray(pr); pr[pr<0]=0
    print('7) predict done, saving...', flush=True)
    idx=np.array(ix[config.dataset_config['group_ids']])
    p=pd.DataFrame(np.concatenate((pr,idx),axis=1),columns=dates+config.dataset_config["group_ids"])
    p=p.melt(id_vars=config.dataset_config['group_ids'],value_vars=dates,var_name='dt',value_name='prediction')
    p[['store_id','product_id']]=p[['store_id','product_id']].astype(int)
    s=p[['store_id','product_id','dt','prediction']].copy()
    s['model_name'],s['input_version'],s['split']='TFT_baseline',iv,'test'
    s.to_parquet(f'{PRED_DIR}/TFT_baseline__{iv}.parquet')
    print(f'8) saved {len(s)} -> TFT_baseline__{iv}.parquet', flush=True)