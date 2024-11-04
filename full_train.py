import os
import cv2, torch, random, itertools, time,datetime
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import transforms
from torch.nn import BCEWithLogitsLoss, MSELoss

from segFormer import *
from dataset import *
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
seed = 2023 
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

def train():
    base_lr = 0.0001
    num_classes = 2
    batch_size = 12
    base_dir = './data/skin/'     # polyp  skin   idrid
    dataset = 'skin'
    image_size = (512,512)
    train_num = 900
    max_epoch = 62    
   
    model = SegFormer_B4(image_size[0],num_classes)
    model.cuda()
   
    model_name = 'SegB4_our_idrid'
    
    print('Total model parameters: ', sum(p.numel() for p in model.parameters())/1e6,'M' )
    print('Trainable model parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad == True)/1e6,'M' )
    
    db_train = MyDataSet(base_dir+'train/', 'train.txt',train_num,image_size,dataset)
    total_samples = len(db_train)
    print("=> Total samples is: {}, labeled samples is: {}".format(total_samples, total_samples))
    train_loader = DataLoader(db_train, batch_size=batch_size,num_workers=4, pin_memory=True)
    print('=> train len:', len(train_loader))

    #optimizer = optim.Adam(model.parameters(), betas=(0.9,0.99), lr=base_lr, weight_decay=0.0001)
    optimizer = optim.AdamW(model.parameters(),lr=base_lr,weight_decay=0.0001)

    ce_loss = CrossEntropyLoss()
    criterion_cos = CriterionCosineSimilarity()
    dice_loss = DiceLoss(num_classes)
    scaler = torch.cuda.amp.GradScaler()

    iter_num = 0
    max_indicator = 0
    best_IoU =0
    best_Dice =0
    best_MAE = 0
    best_Acc = 0
    max_iterations =  max_epoch * len(train_loader)
    for epoch_num in range(max_epoch):
        train_acc = 0
        train_loss = 0
        start_time = time.time()
        model.train()
        
        for  batch_images, batch_labels  in train_loader:
            images, labels = batch_images.cuda(), batch_labels.cuda()
            with torch.cuda.amp.autocast():
                outputs = model(images)
                outputs_soft = torch.softmax(outputs, dim=1)

                ce_losses = ce_loss(outputs,labels.long())
                dice_losses = dice_loss(outputs_soft,labels.long())
                loss = 0.5*(ce_losses+dice_losses) 

                prediction = torch.max(outputs,1)[1]
                train_correct = (prediction == labels).float().mean().cpu().numpy()
                train_acc = train_acc + train_correct
                train_loss = train_loss + loss.detach().cpu().numpy()

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update() 

            lr = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            iter_num = iter_num + 1
                
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        model.eval()
        print('Epoch: {} / {} '.format(epoch_num, max_epoch), 'Training time {}'.format(total_time_str),'Initial LR {:4f}'.format(lr)) 
        print('train_loss: ',train_loss/len(train_loader),' train_acc: ',train_acc/(len(train_loader)),'LR {:4f}'.format(lr)) 
        
        
        save_dir='./results/'
        db_val = testBaseDataSets(base_dir+'test/', 'test.txt',image_size,dataset)
        valloader = DataLoader(db_val, batch_size=1, shuffle=False,num_workers=0)
        model.eval()
        j = 0
        if dataset == 'idrid':
            evaluator_EX= Evaluator()
            evaluator_HE= Evaluator()
            evaluator_MA= Evaluator()
            evaluator_SE= Evaluator()
            with torch.no_grad():
                for sampled_batch in valloader:
                    images, labels = sampled_batch['image'], sampled_batch['label']
                    images, labels = images.cuda(),labels.cuda()

                    pred = model(images)
                    probs = pred.cpu().numpy()
                    save_results(probs,save_dir,image_size[0],image_size[1],j)
                    j = j+1
                    predictions = torch.argmax(pred, dim=1)
                    pred = F.one_hot(predictions.long(), num_classes=num_classes)
                    new_labels =  F.one_hot(labels.long(), num_classes=num_classes)

                    evaluator_EX.update(pred[0,:,:,1], new_labels[0,:,:,1].float())
                    evaluator_HE.update(pred[0,:,:,2], new_labels[0,:,:,2].float())
                    evaluator_MA.update(pred[0,:,:,3], new_labels[0,:,:,3].float())
                    evaluator_SE.update(pred[0,:,:,4], new_labels[0,:,:,4].float())
            MAE_ex, Recall_ex, Pre_ex, Acc_ex, Dice_ex, IoU_ex = evaluator_EX.show(False)
            MAE_he, Recall_he, Pre_he, Acc_he, Dice_he, IoU_he = evaluator_HE.show(False)
            MAE_ma, Recall_ma, Pre_ma, Acc_ma, Dice_ma, IoU_ma = evaluator_MA.show(False)
            MAE_se, Recall_se, Pre_se, Acc_se, Dice_se, IoU_se = evaluator_SE.show(False)
            MAE =  (MAE_ex + MAE_he + MAE_ma +MAE_se )/4
            Acc =  (Acc_ex + Acc_he + Acc_ma +Acc_se )/4
            Dice =  (Dice_ex + Dice_he + Dice_ma +Dice_se)/4 
            IoU =  (IoU_ex + IoU_he + IoU_ma +IoU_se)/4 
            indicator = Dice+IoU  
            if indicator > max_indicator:
                best_Dice =Dice
                best_IoU =IoU
                best_MAE = MAE
                best_Acc = Acc
                max_indicator = indicator
                torch.save(model.state_dict(), './new/'+model_name+'.pth')  
            print("MAE: ", "%.2f" % MAE," Acc: ", "%.2f" % Acc," Dice: ", "%.2f" % Dice," IoU: " , "%.2f" % IoU)         

        else:
            evaluator = Evaluator()
            evaluator2 = Evaluator()
            with torch.no_grad():
                for sampled_batch in valloader:
                    images, labels = sampled_batch['image'], sampled_batch['label']
                    images, labels = images.cuda(),labels.cuda()

                    predictions  = model(images)
                    pred = predictions[0,1,:,:]
                    evaluator.update(pred, labels[0,:,:].float())

                    for i in range(1):
                        #images = images[i].cpu().numpy()
                        #labels = labels.cpu().numpy()
                        #label = (labels[i]*255)
                        pred = pred.cpu().numpy()
                        #cv2.imwrite(save_dir+'image'+str(j)+'.jpg',images.transpose(1, 2, 0)[:,:,::-1])
                        #cv2.imwrite(save_dir+'GT'+str(j)+'.jpg',label*255)
                        cv2.imwrite(save_dir+'Pre'+str(j)+'.jpg',pred*255)
                        j=j+1
            MAE, Recall, Pre, Acc, Dice, IoU = evaluator.show(False)  
            indicator = Dice+IoU  
            if indicator > max_indicator:
                best_Dice =Dice
                best_IoU =IoU
                best_MAE = MAE
                best_Acc = Acc
                max_indicator = indicator
                torch.save(model.state_dict(), './new/'+model_name+'.pth') 
            print("MAE: ", "%.2f" % MAE," Acc: ", "%.2f" % Acc," Dice: ", "%.2f" % Dice," IoU: " , "%.2f" % IoU)

    print('best metric: MAE %.2f Acc %.2f Dice %.2f  IoU %.2f' %(best_MAE,best_Acc,best_Dice, best_IoU))
    with open("./seg.txt","a") as f:
        txt = ''+'MAE:'+str(best_MAE)+' Acc:'+str(best_Acc)+' Dice:'+str(best_Dice)+' IoU:'+str(best_IoU)
        f.write(txt+'\n') 
train()