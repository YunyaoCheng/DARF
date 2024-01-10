import argparse
import torch
import copy
import time
import os
import numpy as np
import torch.optim as optim
from torch.autograd import grad
import logging
from tqdm import tqdm
from model import CORF,Domain_Classifier
import util
from util import set_requires_grad
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

length=24
parser = argparse.ArgumentParser()
parser.add_argument('--adjdata',type=str,default='data/sensor_graph/adj_mx.pkl',help='adj data path')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--gcn_bool',action='store_true',default=True, help='whether to add graph convolution layer')
parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
parser.add_argument('--addaptadj',action='store_true',default=True,help='whether add adaptive adj')
parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
parser.add_argument('--seq_length',type=int,default=length,help='')
parser.add_argument('--nhid',type=int,default=length,help='')
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=7,help='number of nodes')


parser.add_argument('--nb_blocks_per_stack',type=int,default=2)
parser.add_argument('--forecast_length',type=int,default=length)
parser.add_argument('--backcast_length',type=int,default=length)
parser.add_argument('--share_weights_in_stack',action='store_true',default=False)
parser.add_argument('--hidden_layer_units',type=int,default=256)

parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data_path',type=str,default='data/Anomaly_ETT.csv')
parser.add_argument('--cal_data_path',type=str, default='data/Normal_ETT.csv')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=100,help='')
parser.add_argument('--re_epochs',type=int,default=10,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
parser.add_argument('--expid',type=int,default=1,help='experiment id')
parser.add_argument('--load_model',action='store_true', default=False,help='whether to load model')
parser.add_argument('--model_path',type=str, default='./result/')
parser.add_argument('--omega',type=float,default=0.5)
parser.add_argument('--lambda',type=float,default=1.0)
parser.add_argument('--log_dir',type=str, default='train.log')
parser.add_argument('--is_training', type=int, default=1, help='status')
args = parser.parse_args()


def gradient_penalty(critic, h_s, h_t):
    alpha = torch.rand(h_s.size(0), 1).to(args.device)
    differences = h_t - h_s
    interpolates = h_s + (alpha * differences)
    interpolates = torch.stack([interpolates, h_s, h_t]).requires_grad_()

    preds = critic(interpolates)
    gradients = grad(preds, interpolates,
                     grad_outputs=torch.ones_like(preds),
                     retain_graph=True, create_graph=True)[0]
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1)**2).mean()
    return gradient_penalty


def pre_train(model, optimizer, dataloader, model_path):
    device = torch.device(args.device)
    scaler = dataloader['scaler']

    loss_func = util.masked_mae
    clip = 5
    model.train()
    train_time=[]
    total_step = args.batch_size * dataloader['train_loader'].get_num_batch()
    best_mae =9999999999
    logging.info("start to train..............................")
    for epoch in range(args.epochs+1):
        CORF_loss = []
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        start_step = epoch * dataloader['train_loader'].get_num_batch()
        for iter, (x, y) in tqdm(enumerate(dataloader['train_loader'].get_iterator())):
            p = float(iter + start_step) / total_step
            constant = 2. / (1. + np.exp(-10 * p)) - 1

            trainx = torch.FloatTensor(x).to(device)
            trainx = trainx.transpose(1, 3)
            trainy = torch.FloatTensor(y).to(device)
            trainy = trainy.transpose(1, 3)


            optimizer.zero_grad()

            embedding, output = model(trainx)
            real = trainy
            predict = output

            loss = loss_func(predict, real, 0.0)
            loss.backward()
            # if clip is not None:
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            mape = util.masked_mape(predict, real, 0.0).item()
            rmse = util.masked_rmse(predict, real, 0.0).item()

            CORF_loss.append(loss_func(predict, real, 0.0).item())
            train_loss.append(loss.item())
            train_mape.append(mape)
            train_rmse.append(rmse)

        t2 = time.time()
        train_time.append(t2 - t1)

        corf_loss = np.mean(CORF_loss)
        logging.info('Epoch: {:03d},CORF_loss:{:.4f}'.format(epoch, corf_loss))


        #Validation
        model.eval()
        val_mape=[]
        val_rmse=[]
        val_loss=[]
        #logging.info("start val----------------------------")
        for i, (valx, valy) in tqdm(enumerate(dataloader['val_loader'].get_iterator())):
            valx = torch.tensor(valx).to(args.device)
            valy = torch.tensor(valy).to(args.device)
            valx = valx.transpose(1,3)
            valy = valy.transpose(1,3)


            embedding, val_pred = model(valx)
            #val_pred = val_pred.transpose(1, 3)
            #real = torch.unsqueeze(valy, dim=1)
            predict = val_pred
            real = valy
            #predict = scaler.inverse_transform(val_pred)
            loss = loss_func(predict,real, 0.0).item()
            mape = util.masked_mape(predict, real, 0.0).item()
            rmse = util.masked_rmse(predict, real, 0.0).item()
            val_loss.append(loss)
            val_mape.append(mape)
            val_rmse.append(rmse)

        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)
        mvalid_loss = np.mean(val_loss)
        mvalid_mape = np.mean(val_mape)
        mvalid_rmse = np.mean(val_rmse)
        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}'
        logging.info(
            log.format(epoch, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse))


        if mvalid_loss< best_mae:
            best_mae = mvalid_loss
            best_model = model
            ## save model
            state = dict([('model', copy.deepcopy(model.state_dict())),
                          ('optim', copy.deepcopy(optimizer.state_dict()))])
            util.Save_Model(state,epoch,model_path)
            logging.info("save model from epoch {}".format(epoch))
    return best_model


def re_train(model, optimizer, optimizer_domain, domain_classifier, dataloader, source_dataloader, target_dataloader, model_path):
    logging.info(f'Loading pretrained model')
    state = util.Load_Model(model_path)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optim'])
    logging.info("load pretrained model successfully")

    device = torch.device(args.device)
    scaler = dataloader['scaler']

    loss_func = util.masked_mae
    clip = 5
    model.train()
    domain_classifier.train()
    train_time=[]
    total_step = args.batch_size * dataloader['train_loader'].get_num_batch()
    best_mae =9999999999
    logging.info("start to train..............................")
    for epoch in range(args.re_epochs+1):
        Loss = []
        t1 = time.time()
        start_step = epoch * dataloader['train_loader'].get_num_batch()

        for ((source_trainx, source_trainy), (target_trainx, target_trainy)) in zip(source_dataloader['train_loader'].get_iterator(), target_dataloader['train_loader'].get_iterator()):

            set_requires_grad(model, requires_grad=False)
            set_requires_grad(domain_classifier, requires_grad=True)

            source_trainx = torch.FloatTensor(source_trainx).to(device)
            source_trainx = source_trainx.transpose(1, 3)
            source_trainy = torch.FloatTensor(source_trainy).to(device)
            source_trainy = source_trainy.transpose(1, 3)

            target_trainx = torch.FloatTensor(target_trainx).to(device)
            target_trainx = target_trainx.transpose(1, 3)
            target_trainy = torch.FloatTensor(target_trainy).to(device)
            target_trainy = target_trainy.transpose(1, 3)

            optimizer_domain.zero_grad()
            source_embedding, source_output = model(source_trainx)
            target_embedding, target_output = model(target_trainx)

            # train domain_classifer
            source_embedding = source_embedding.reshape(args.batch_size, -1).to(args.device)
            source_domain_pred = domain_classifier(source_embedding)
            target_embedding = target_embedding.reshape(args.batch_size, -1).to(args.device)
            target_domain_pred = domain_classifier(target_embedding)
            gp = gradient_penalty(domain_classifier, source_embedding, target_embedding)

            wasserstein_distance = source_domain_pred.mean() - target_domain_pred.mean()
            domain_loss = -wasserstein_distance + args.omega*gp
            domain_loss.backward()
            optimizer_domain.step()


            set_requires_grad(model, requires_grad=True)
            set_requires_grad(domain_classifier, requires_grad=False)

            optimizer.zero_grad()
            source_embedding, source_output = model(source_trainx)
            target_embedding, target_output = model(target_trainx)

            source_real = source_trainy
            target_real = target_trainy
            source_predict = source_output
            target_predict = target_output

            # train domain_classifer
            source_embedding = source_embedding.reshape(args.batch_size, -1).to(args.device)
            source_domain_pred = domain_classifier(source_embedding)
            target_embedding = target_embedding.reshape(args.batch_size, -1).to(args.device)
            target_domain_pred = domain_classifier(target_embedding)
            wasserstein_distance = source_domain_pred.mean() - target_domain_pred.mean()

            loss = loss_func(source_predict, source_real, 0.0) + loss_func(target_predict, target_real, 0.0) + wasserstein_distance*1.0
            loss.backward()
            optimizer.step()
            Loss.append(loss.item())


        t2 = time.time()
        train_time.append(t2 - t1)

        Loss = np.mean(Loss)


        #Validation
        model.eval()
        domain_classifier.eval()
        val_mape=[]
        val_rmse=[]
        val_loss=[]
        #logging.info("start val----------------------------")
        for i, (valx, valy) in tqdm(enumerate(dataloader['val_loader'].get_iterator())):
            valx = torch.tensor(valx).to(args.device)
            valy = torch.tensor(valy).to(args.device)
            valx = valx.transpose(1,3)
            valy = valy.transpose(1,3)


            embedding, val_pred = model(valx)
            #val_pred = val_pred.transpose(1, 3)
            #real = torch.unsqueeze(valy, dim=1)
            predict = val_pred
            real = valy
            #predict = scaler.inverse_transform(val_pred)
            loss = loss_func(predict,real, 0.0).item()
            mape = util.masked_mape(predict, real, 0.0).item()
            rmse = util.masked_rmse(predict, real, 0.0).item()
            val_loss.append(loss)
            val_mape.append(mape)
            val_rmse.append(rmse)

        mvalid_loss = np.mean(val_loss)
        mvalid_mape = np.mean(val_mape)
        mvalid_rmse = np.mean(val_rmse)
        log = 'Epoch: {:03d}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}'
        logging.info(
            log.format(epoch, mvalid_loss, mvalid_mape, mvalid_rmse))


        if mvalid_loss< best_mae:
            best_mae = mvalid_loss
            best_model = model
            ## save model
            state = dict([('model', copy.deepcopy(model.state_dict())),
                          ('optim', copy.deepcopy(optimizer.state_dict())),
                          ('domain_classifier', copy.deepcopy(domain_classifier.state_dict()))])
            util.Save_Model(state,epoch,model_path)
            logging.info("save model from epoch {}".format(epoch))
    return best_model


def predict(model, optimizer,dataloader):
    state = util.Load_Model(model_path)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optim'])
    model.eval()
    logging.info('start to predict-----------------------------------------------')
    test_mae=[]
    test_rmse=[]
    test_mape=[]
    outputs = []
    for i,(testx, testy) in tqdm(enumerate(dataloader['test_loader'].get_iterator())):
        testx = torch.tensor(testx).to(args.device)
        testy = torch.tensor(testy).to(args.device)
        testx = testx.transpose(1,3)
        testy = testy.transpose(1,3)
        backward, test_pred = model(testx)
        mae = util.masked_mae(test_pred, testy, 0.0).item()
        mape = util.masked_mape(test_pred, testy, 0.0).item()
        rmse = util.masked_rmse(test_pred, testy, 0.0).item()
        test_mape.append(mape)
        test_rmse.append(rmse)
        test_mae.append(mae)
        outputs.append(test_pred.squeeze())

    mtest_mae = np.mean(test_mae)
    mtest_mape = np.mean(test_mape)
    mtest_rmse = np.mean(test_rmse)
    logging.info('Test MAE:{:.4f}, Test MAPE:{:.4f}, Test RMSE:{:.4f}'.format(mtest_mae, mtest_mape, mtest_rmse))




util.set_logger(args.log_dir)
print(args)


#domain_criterion = torch.nn.CrossEntropyLoss()
domain_classifier = Domain_Classifier(num_class=1, encode_dim=args.seq_length*args.seq_length*args.num_nodes)
domain_classifier = domain_classifier.to(args.device)


supports=None
aptinit = None
cur_dir = os.getcwd()
args.load_model=False
logging.info(f'\n\n****************************************************************************************************************')
logging.info("Begin")
logging.info(f'****************************************************************************************************************\n\n')

if args.seq_length>12:
    kernel_size=3
    layers=2
    blocks=math.ceil(args.seq_length/6)
else:
    kernel_size=2
    layers=2
    blocks=math.ceil(args.seq_length/3)

model = CORF(args.device, args.num_nodes, args.dropout, supports, args.gcn_bool, args.addaptadj,
                aptinit, args.in_dim, args.seq_length,
                args.nhid, kernel_size, blocks, layers, args.forecast_length, args.backcast_length, args.nb_blocks_per_stack)
optimizer = optim.Adam([{'params': model.parameters()}],
                       lr=args.learning_rate, weight_decay=args.weight_decay)
optimizer_domain = optim.Adam([{'params': domain_classifier.parameters()}],
                       lr=args.learning_rate*0.01, weight_decay=args.weight_decay)
model_path = os.path.join('{}'.format(cur_dir), 'result')
if args.load_model:
    logging.info(f'Loading pretrained model')
    state = util.Load_Model(model_path)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optim'])
    domain_classifier.load_state_dict(state['domain_classifier'])
    logging.info("load pretrained model successfully")
else:
    logging.info(f'No existing pretrained model at {model_path}')

source_dataloader = util.load_source_dataset(args.data_path,args.cal_data_path,args.seq_length,args.batch_size,args.batch_size, args.batch_size)
target_dataloader = util.load_target_dataset(args.data_path,args.cal_data_path,args.seq_length,args.batch_size,args.batch_size, args.batch_size)
all_dataloader = util.load_all_dataset(args.data_path,args.cal_data_path,args.seq_length,args.batch_size,args.batch_size, args.batch_size)
if args.is_training:
    pre_train(model, optimizer, source_dataloader, model_path)
    re_train(model, optimizer, optimizer_domain, domain_classifier, all_dataloader, source_dataloader, target_dataloader, model_path)
predict(model,optimizer,all_dataloader)
