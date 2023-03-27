import os
import torch
import torch.nn.functional as F
import sys
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from Code.lib.model import SPNet
from Code.utils.Dataloader import get_dataloader
from Code.utils.utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from Code.utils.options import opt

import torch.nn as nn

#set the device for training
if opt.gpu_id=='0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')

  
cudnn.benchmark = True

#build the model
model = SPNet(32,50)
if(opt.load is not None):
    model.load_state_dict(torch.load(opt.load))
    print('load model from ',opt.load)
    
model.cuda()
params    = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)


#set the path
save_path        = opt.save_path


if not os.path.exists(save_path):
    os.makedirs(save_path)

#load data
print('load data...')
train_Dataloader, val_Dataloader = get_dataloader()


logging.basicConfig(filename=save_path+'log.log',format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level = logging.INFO,filemode='a',datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("BBSNet_unif-Train")
logging.info("Config")
logging.info('epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(opt.epoch,opt.lr,opt.batchsize,opt.trainsize,opt.clip,opt.decay_rate,opt.load,save_path,opt.decay_epoch))


step = 0
writer     = SummaryWriter(save_path+'summary')
best_mae   = 1
best_epoch = 0


#set loss function
l1Loss = nn.L1Loss()
l2Loss = nn.MSELoss()



def train(train_loader, model, optimizer, epoch,save_path):
    global step
    model.train()
    loss_all=0
    epoch_step=0
    total_step = len(train_loader)


    for i, (images, gts,_) in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        
        images   = images.cuda()
        gts      = gts.cuda()

        ##
        pre_res  = model(images)
        
        loss1    = l1Loss(pre_res,gts) 
        loss2    = l2Loss(pre_res,gts) 
        
        loss_seg = loss1 + loss2 

        loss = loss_seg 
        loss.backward()

        clip_gradient(optimizer, opt.clip)
        optimizer.step()
        step+=1
        epoch_step+=1
        loss_all+=loss.data
        if i % 50 == 0 or i == total_step or i==1:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f} Loss2: {:0.4f}'.
                format(datetime.now(), epoch, opt.epoch, i, total_step, loss1.data, loss2.data))
            logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f} Loss2: {:0.4f}'.
                format( epoch, opt.epoch, i, total_step, loss1.data, loss2.data))
            
    loss_all/=epoch_step
    logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format( epoch, opt.epoch, loss_all))
    writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
    
        
#test function
def val(test_loader,model,epoch,save_path):
    global best_mae,best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum=0
        for i, (images, gts,_) in enumerate(test_loader, start=1):
            images   = images.cuda()
            pre_res = model(images)
            res     = pre_res.sigmoid().data.cpu().numpy()
            res     = (res - res.min()) / (res.max() - res.min() + 1e-8)
            gts     = gts.numpy()
            mae_sum += np.sum(np.abs(res-gts))
            
        mae = mae_sum/len(test_loader)
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        if epoch==1:
            best_mae = mae
            torch.save(model.state_dict(), save_path+'SPNet_epoch_best.pth')
        else:
            if mae<best_mae:
                best_mae   = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path+'SPNet_epoch_best.pth')
                print('best epoch:{}'.format(epoch))
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch,mae,best_mae,best_epoch))

        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch,mae,best_epoch,best_mae))
 
if __name__ == '__main__':
    print("Start train...")
    
    for epoch in range(1, opt.epoch+1):
        
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        # train
        train(train_Dataloader, model, optimizer, epoch,save_path)
        
        #test
        val(val_Dataloader,model,epoch,save_path)
