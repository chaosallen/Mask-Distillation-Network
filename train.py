import torch
import torch.nn as nn
import numpy as np
import logging
import sys
import os
import model
import shutil
from options.train_options import TrainOptions
import Batchreader
import natsort
import scipy.misc as misc
import imageio
import focalloss
import Kappa

def train_net(net,device):
    #train setting
    interval=opt.save_interval
    best_valid_s = 0
    best_valid_t = 0
    model_save_path = os.path.join(opt.saveroot, 'checkpoints')
    best_model_save_path = os.path.join(opt.saveroot, 'best_model')
    Trainreader=Batchreader.Batchreader(opt.dataroot)
    # Setting Optimizer
    optimizer = torch.optim.Adam(net.parameters(), opt.lr, betas=(0.9, 0.99))
    #Setting Loss
    CE = nn.CrossEntropyLoss()
    FL = focalloss.FocalLoss()
    L1 = nn.L1Loss()
    #Start train
    for itr in range(0, opt.max_iteration):
        net.train()
        train_student,train_teacher, train_annotations = Trainreader.next_batch(opt.batch_size,opt.input_size)
        train_student = train_student.to(device=device, dtype=torch.float32)
        train_teacher = train_teacher.to(device=device, dtype=torch.float32)
        train_annotations = train_annotations.to(device=device, dtype=torch.long)
        if opt.Network_mode== 'S':
            pred,_ = net(train_student)
            loss = CE(pred, train_annotations)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if opt.Network_mode== 'T':
            pred,_ = net(train_teacher)
            loss = CE(pred, train_annotations)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if opt.Network_mode== 'ST' :
            preds,predt,cams,camt = net(train_student,train_teacher)
            loss_s = CE(preds, train_annotations)
            loss_t = CE(predt, train_annotations)
            loss_cc = L1(preds,predt)
            loss_ac = L1(cams,camt)
            loss=loss_s+loss_t+loss_ac+loss_cc
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        if itr % 10 == 0:
            print(itr,loss.item())
        #Start Val
        with torch.no_grad():
            if itr % interval==0:
                #Save model
                torch.save(net.module.state_dict(),
                           os.path.join(model_save_path,f'{itr}.pth'))
                logging.info(f'Checkpoint {itr} saved !')
                #Calculate validation acc
                net.eval()
                num = 0
                studentmetric = np.zeros([4, 4])
                teachermetric = np.zeros([4, 4])
                images_student = np.zeros((1, opt.input_size[2], opt.input_size[0], opt.input_size[1]))
                images_teacher = np.zeros((1, opt.input_size[2], opt.input_size[0], opt.input_size[1]))
                resultlists = []
                resultlistt = []
                gtlist = []
                for classnum in range(1, 5):
                    dirs = os.listdir(os.path.join(opt.dataroot, 'Val', 'S', str(classnum)))
                    dirs = natsort.natsorted(dirs)
                    for testfile in dirs:
                        images_student[0, :, :, :] =np.array(imageio.imread(os.path.join(opt.dataroot, 'Val', 'S', str(classnum), testfile))).transpose(2, 0, 1)
                        images_teacher[0, :, :, :] =np.array(imageio.imread(os.path.join(opt.dataroot, 'Val', 'T', str(classnum), testfile))).transpose(2, 0, 1)
                        images_s = torch.from_numpy(images_student)
                        images_t = torch.from_numpy(images_teacher)
                        images_s = images_s.to(device=device, dtype=torch.float32)
                        images_t = images_t.to(device=device, dtype=torch.float32)
                        gtlist.append(classnum - 1)
                        num += 1
                        if opt.Network_mode == 'S':
                            preds,_ = net(images_s)
                            preds = torch.argmax(preds, dim=1)
                            results = preds.cpu().detach().numpy()[0]
                            studentmetric[results, classnum - 1] += 1
                            resultlists.append(results)
                        if opt.Network_mode == 'T':
                            predt,_ = net(images_t)
                            predt = torch.argmax(predt, dim=1)
                            resultt = predt.cpu().detach().numpy()[0]
                            teachermetric[resultt, classnum - 1] += 1
                            resultlistt.append(resultt)
                        if opt.Network_mode == 'ST':
                            preds,predt, _, _ = net(images_s, images_t)
                            preds = torch.argmax(preds, dim=1)
                            predt = torch.argmax(predt, dim=1)
                            results = preds.cpu().detach().numpy()[0]
                            resultt = predt.cpu().detach().numpy()[0]
                            studentmetric[results, classnum - 1] += 1
                            teachermetric[resultt, classnum - 1] += 1
                            resultlists.append(results)
                            resultlistt.append(resultt)

                if opt.Network_mode == 'S' or opt.Network_mode =='ST':
                    accs = (studentmetric[0, 0] + studentmetric[1, 1] + studentmetric[2, 2] + studentmetric[3, 3]) / num
                    kappas = Kappa.quadratic_weighted_kappa(resultlists, gtlist)
                    print(studentmetric)
                    print("Step:{}, Valid_Acc:{}".format(itr, accs))
                    print("Step:{}, Valid_Kappa:{}".format(itr, kappas))
                    temps = round(accs, 6)
                    # save best model
                    if temps > best_valid_s:
                        os.mkdir(os.path.join(best_model_save_path, 'S', str(temps)))
                        temp2 = f'{itr}.pth'
                        shutil.copy(os.path.join(model_save_path, temp2),
                                    os.path.join(best_model_save_path, 'S', str(temps), temp2))
                        model_names = natsort.natsorted(os.listdir(os.path.join(best_model_save_path, 'S')))
                        # print(len(model_names))
                        if len(model_names) == 4:
                            shutil.rmtree(os.path.join(best_model_save_path, 'S', model_names[0]))
                        best_valid_s = temps
                if opt.Network_mode == 'T' or opt.Network_mode == 'ST':
                    acct = (teachermetric[0, 0] + teachermetric[1, 1] + teachermetric[2, 2] + teachermetric[3, 3]) / num
                    kappat = Kappa.quadratic_weighted_kappa(resultlistt, gtlist)
                    print(teachermetric)
                    print("Step:{}, Valid_Acc:{}".format(itr, acct))
                    print("Step:{}, Valid_Kappa:{}".format(itr, kappat))
                    tempt = round(acct, 6)
                    # save best model
                    if tempt > best_valid_t:
                        os.mkdir(os.path.join(best_model_save_path, 'T', str(tempt)))
                        temp2 = f'{itr}.pth'
                        shutil.copy(os.path.join(model_save_path, temp2),
                                    os.path.join(best_model_save_path, 'T', str(tempt), temp2))
                        model_names = natsort.natsorted(os.listdir(os.path.join(best_model_save_path, 'T')))
                        # print(len(model_names))
                        if len(model_names) == 4:
                            shutil.rmtree(os.path.join(best_model_save_path, 'T', model_names[0]))
                        best_valid_t = tempt



if __name__ == '__main__':
    #setting logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    #loading options
    opt = TrainOptions().parse()
    #setting GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    #loading network
    if opt.Network_mode != 'ST':
        net = model.subnet(in_channels=opt.in_channels, n_classes=opt.n_classes)
    else:
        net = model.MDN(in_channels=opt.in_channels, n_classes=opt.n_classes)
    net=torch.nn.DataParallel(net,[0]).cuda()
    print('parameters:',sum(param.numel() for param in net.parameters()))
    #load trained model
    if opt.load:
        net.load_state_dict(
            torch.load(opt.load, map_location=device)
        )
        logging.info(f'Model loaded from {opt.load}')
    try:
        train_net(net=net,device=device)
    except KeyboardInterrupt:
        #torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)




