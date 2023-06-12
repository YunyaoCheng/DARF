import pandas as pd
import numpy as np
from datetime import timedelta
import argparse
import os


def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=False, add_day_in_week=False, scaler=None
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes = df.shape
    df.index = pd.to_datetime(df["Date"])
    data = np.expand_dims(df.values, axis=-1)
    feature_list = [data]
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(time_in_day)
    if add_day_in_week:
        dow = df.index.dayofweek
        dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)

    data = np.concatenate(feature_list, axis=-1)
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):  # t is the index of the last observation.
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y

def seq_is_holiday(X,y):
    """
    Check to see if a subsequence contains holidays, if it contains holidays in the holidays, if it does not include holidays, it is placed in non-holidays
    :param x:
    :param y:
    :return:
    """

    N = X.shape[0]
    M = X.shape[1]

    holiday = list()
    no_holiday = list()

    for i in range(N):
        if (X[i,:,-1,0] == np.zeros(M)).all():
            no_holiday.append(i)
        else:
            holiday.append(i)
    holiday_X = X[holiday,:,:,:]
    holiday_y = y[holiday,:,:,:]
    no_holiday_X = X[no_holiday,:,:,:]
    no_holiday_y = y[no_holiday,:,:,:]
    return holiday_X[:,:,1:-1,:], no_holiday_X[:,:,1:-1,:],holiday_y[:,:,1:-1,:], no_holiday_y[:,:,1:-1,:]



def balance_data(data1, data2):
    N1,N2 = data1.shape[0],data2.shape[0]
    if N1>N2:
        num1 = int(N1/N2)
        num2 = N1%N2
        data2 = data2.repeat(num1,axis=0)
        data2 = np.concatenate([data2,data2[:num2,:,:,:]])
    else:
        num1 = int(N2/N1)
        num2 = N2%N1
        data1 = data1.repeat(num1,axis=0)
        data1 = np.concatenate([data1, data1[:num2,:,:,:]])
    return data1, data2



def generate_train_val_test(df, cal_df,seq_length_x,
                            seq_length_y,y_start,
                            add_time_in_day,
                            add_day_in_week):
    seq_length_x, seq_length_y = seq_length_x, seq_length_y

    # 0 is the latest observed sample.
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    # Predict the next one hour
    y_offsets = np.sort(np.arange(y_start, (seq_length_y + 1), 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)


    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=add_time_in_day,
        add_day_in_week=add_day_in_week,
    )

    cal_x, cal_y = generate_graph_seq2seq_io_data(
        cal_df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=add_time_in_day,
        add_day_in_week=add_day_in_week,
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    # Write the data into npz file.

    

    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train
    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    x_test, y_test = x[-num_test:], y[-num_test:]
    print("------------------")
    print(x_test.shape,y_test.shape)
    print(x_val.shape, y_val.shape)
    print("------------------")

    cal_x_train, cal_y_train = cal_x[:num_train], cal_y[:num_train]

   

    holiday_x_train, no_holiday_x_train, holiday_y_train, no_holiday_y_train = seq_is_holiday(x_train, y_train)
    holiday_x_test, no_holiday_x_test, holiday_y_test,no_holiday_y_test = seq_is_holiday(x_test, y_test)
    holiday_x_val, no_holiday_x_val, holiday_y_val, no_holiday_y_val = seq_is_holiday(x_val, y_val)



    #balance holiday data and no_holiday data
    holiday_x_train, no_holiday_x_train = balance_data(holiday_x_train, no_holiday_x_train)
    holiday_y_train, no_holiday_y_train = balance_data(holiday_y_train, no_holiday_y_train)

    holiday = dict()
    no_holiday = dict()
    holiday['x_train'] = holiday_x_train
    holiday['y_train'] = holiday_y_train
    holiday['x_test'] = holiday_x_test
    holiday['y_test'] = holiday_y_test
    holiday['x_val'] = holiday_x_val
    holiday['y_val'] = holiday_y_val
    no_holiday['x_train'] = no_holiday_x_train
    no_holiday['y_train'] = no_holiday_y_train
    no_holiday['x_test'] = no_holiday_x_test
    no_holiday['y_test'] = no_holiday_y_test
    no_holiday['x_val'] = no_holiday_x_val
    no_holiday['y_val'] = no_holiday_y_val
    return holiday, no_holiday,cal_x_train[:,:,1:-1,:]


