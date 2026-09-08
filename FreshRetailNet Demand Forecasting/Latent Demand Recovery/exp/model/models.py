import os
import torch
from torch import nn
from pypots.imputation import TimesNet
from pypots.optim import Adam
from pypots.utils.logging import logger

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(DEVICE)

def load_model(CONFIG):
    model = CONFIG.get('model', 'TimesNet')
    saving_path = os.path.join(CONFIG['saving_path'], model)
    if model == 'TimesNet':
        model = TimesNet(
            n_steps = CONFIG['n_steps'],
            n_features = CONFIG['n_features'],
            n_layers = CONFIG['n_layers'],
            top_k = 7,
            d_model = CONFIG['d_model'],
            d_ffn = CONFIG['d_ffn'],
            n_kernels = 5,
            dropout = CONFIG['dropout'],
            apply_nonstationary_norm = True,
            epochs = CONFIG['EPOCHS'],
            batch_size = CONFIG['batch_size'],
            saving_path = saving_path,
            optimizer = Adam(lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay']),
            device = DEVICE,
            patience = CONFIG['patience'],
            OT = CONFIG['OT']
        )
    else:
        raise NotImplementedError(f'{model} is not implemented in this trimmed thesis copy (TimesNet only)')
    return model
