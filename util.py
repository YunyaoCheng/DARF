import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
import datetime
import logging
import pandas as pd
from sklearn import utils
from data_processing import generate_train_val_test
class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, cut_last_batch=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if cut_last_batch:
            num = len(xs) - (len(xs) % batch_size)
            xs = xs[:num].astype(np.float32)
            ys = ys[:num].astype(np.float32)
        # if pad_with_last_sample:
        #     num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
        #     x_padding = np.repeat(xs[-1:], num_padding, axis=0)
        #     y_padding = np.repeat(ys[-1:], num_padding, axis=0)
        #     xs = np.concatenate([xs, x_padding], axis=0)
        #     ys = np.concatenate([ys, y_padding], axis=0)
        #     xs = xs.astype(np.float32)
        #     ys = ys.astype(np.float32)

        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_num_batch(self):
        return self.num_batch

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()

class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean




def load_all_dataset(data_path,cal_data_path,seq_length,
                     batch_size, valid_batch_size=None,
                     test_batch_size=None):
    data = {}
    data_df = pd.read_csv(data_path)

    cal_df = pd.read_csv(cal_data_path)
    #cal_df = cal_df.iloc[:,1:-1]
    holiday_data, no_holiday_data, cal_x_train = generate_train_val_test(data_df,cal_df,seq_length_x=seq_length,
                                                            seq_length_y=seq_length,y_start=1,
                                                            add_time_in_day=False,
                                                            add_day_in_week=False)
    cal_x_train = cal_x_train.astype('float')
    print("cal_x_train:")
    print(cal_x_train.shape)
    for category in ['train', 'val', 'test']:
        data['x_' + category] = np.concatenate([holiday_data['x_'+category].astype(np.float32),no_holiday_data['x_'+category].astype(np.float32)])
        data['y_' + category] = np.concatenate([holiday_data['y_'+category].astype(np.float32),no_holiday_data['y_'+category].astype(np.float32)])
        M, N = holiday_data['x_'+category].shape[0], no_holiday_data['x_'+category].shape[0]
        data[category+'_label'] = np.append(np.ones(M),np.zeros(N))
        data['x_' + category],data['y_'+ category], data[category + '_label'] = utils.shuffle(data['x_' + category],data['y_'+ category], data[category + '_label'])
    #scaler = StandardScaler(mean=data['x_train'][..., 0].mean(axis=0), std=data['x_train'][..., 0].std(axis=0))
    scaler = StandardScaler(mean=cal_x_train[..., 0].mean(axis=0), std=cal_x_train[..., 0].std(axis=0))
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
        data['y_' + category][..., 0] = scaler.transform(data['y_' + category][..., 0])

    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    return data







# def load_dataset(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None):
#     data = {}
#     for category in ['train', 'val', 'test']:
#         cat_data = np.load(os.path.join(dataset_dir, category + '.npz'),allow_pickle=True)
#         data['x_' + category] = cat_data['x'].astype(np.float32)
#         data['y_' + category] = cat_data['y'].astype(np.float32)
#     scaler = StandardScaler(mean=data['x_train'][..., 0].mean(axis=0), std=data['x_train'][..., 0].std(axis=0))
#     # Data format
#     for category in ['train', 'val', 'test']:
#         data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
#         data['y_' + category][..., 0] = scaler.transform(data['y_' + category][..., 0])
#     data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
#     data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
#     data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
#     data['scaler'] = scaler
#     return data




def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)

    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    # print("-------------------------------")
    # print(preds.shape)
    # print(labels.shape)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs((preds-labels)/(labels+4))
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse


def Save_Model(state, epoch,path='result',**kwargs):
    if not os.path.exists(path):
        os.makedirs(path)
    if kwargs.get('name',None) is None:
        cur_time = datetime.datetime.now().strftime('%Y-%m-%d#%H:%M:%S')
        name = cur_time +'--epoch:{}'.format(epoch)
        full_name= os.path.join(path,name)
        torch.save(state, full_name)
        print("Saved model at epoch {} successfully".format(epoch))
        with open('{}/checkpoint'.format(path),'w') as file:
            file.write(name)
            print("Write to checkpoint")

def Load_Model(path='result', **kwargs):
    if kwargs.get('name', None) is None:
        with open('{}/checkpoint'.format(path)) as file:
            content = file.read().strip()
            name = os.path.join(path, content)
    else:
        name=kwargs['name']
        name = os.path.join(path,name)
    state = torch.load(name, map_location=lambda storage, loc: storage)
    print('load model {} successfully'.format(name))
    return state



def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)