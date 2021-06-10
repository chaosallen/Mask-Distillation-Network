import torch
import torch.nn as nn
import logging
import sys
import os
import model
import numpy as np
import scipy.misc as misc
from options.test_options import TestOptions
import natsort
import seaborn as sns
from scipy import io
import matplotlib.pyplot as plt
import imageio
import Kappa
import grad_cam
import time
from PIL import Image


from misc_functions import save_class_activation_images

def test_net(net,device):

    test_results = os.path.join(opt.saveroot, 'test_results')
    net.eval()
    images_student= np.zeros((1, opt.input_size[2], opt.input_size[0], opt.input_size[1]))
    images_teacher = np.zeros((1, opt.input_size[2], opt.input_size[0], opt.input_size[1]))
    resultlist=[]
    gtlist=[]
    namelist=[]
    num=0
    studentmetric = np.zeros([4, 4])
    for classnum in range(1, 5):
        dirs = os.listdir(os.path.join(opt.dataroot,'Test','S',str(classnum)))
        dirs = natsort.natsorted(dirs)
        for testfile in dirs:
            images_student[0, :, :, :] =np.array(imageio.imread(os.path.join(opt.dataroot, 'Test', 'S', str(classnum), testfile))).transpose(2, 0, 1)
            images_teacher[0, :, :, :] =np.array(imageio.imread(os.path.join(opt.dataroot, 'Test', 'T', str(classnum), testfile))).transpose(2, 0, 1)
            images_s = torch.from_numpy(images_student)
            images_t = torch.from_numpy(images_teacher)
            images_s = images_s.to(device=device, dtype=torch.float32)
            images_t = images_t.to(device=device, dtype=torch.float32)
            start_time=time.time()
            if opt.Network_mode=='S':
                pred,_= net(images_s)
            if opt.Network_mode=='T':
                pred,_= net(images_t)
            if opt.Network_mode=='ST':
                pred,_,_,_= net(images_s,images_t)
            end_time=time.time()
            if opt.print_cam:
                gradcam = grad_cam.GradCam(net.teachernet, 1)
                # Generate cam mask
                cam = gradcam.generate_cam(images_t, target_class=classnum-1)
                # Save mask
                image_visual=images_t.squeeze(0).cpu().numpy().transpose(1,2,0)
                image_visual=Image.fromarray(np.uint8(image_visual),'RGB')
                save_class_activation_images(image_visual, cam, os.path.join('/home/limingchao/PycharmProjects/untitled/BJ_Classification_pytorch/LXD_result/gradcam/',str(classnum),testfile))
            pred = torch.argmax(pred, dim=1)
            result=pred.cpu().detach().numpy()[0]
            print(testfile,result,end_time-start_time)
            studentmetric[result, classnum - 1] += 1
            num+=1
            namelist.append(testfile)
            resultlist.append(result)
            gtlist.append(classnum - 1)
    kappa=Kappa.quadratic_weighted_kappa(resultlist,gtlist)
    np.save(os.path.join(opt.saveroot,'results',opt.backbone +'_'+ opt.Network_mode+'.npy'),resultlist)
    np.save(os.path.join(opt.saveroot, 'results', 'namelist.npy'), namelist)
    np.save(os.path.join(opt.saveroot, 'results', 'gtlist.npy'), gtlist)
    acc=(studentmetric[0,0]+studentmetric[1,1]+studentmetric[2,2]+studentmetric[3,3])/num
    print(studentmetric)
    print('acc:',acc)
    print('kappa:',kappa)

    # visual
    ax = sns.heatmap(studentmetric,
                     cmap="Blues",  # 图中的主色调
                     xticklabels=[1, 2, 3, 4],  # 预测标签
                     yticklabels=[1, 2, 3, 4],  # 真实标签
                     linewidths=.5,  # 格子与格子之间的空隙
                     square=True,  # 图是方的
                     fmt="g",  # 图中每个方格数字的格式化方式，g代表完整输出
                     annot=True)  # 允许注释
    # 下面四行是兼容性代码，为了兼容新版的plt
    b, t = plt.ylim()
    b += 0.5
    t -= 0.5
    plt.ylim(b, t)
    # 非Ipython环境要加下面一句
    plt.show()

    # 画出来的图如果种类较多建议把图片放大了看，所有的类别就能清晰隔开


if __name__ == '__main__':
    #setting logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    #loading options
    opt = TestOptions().parse()
    #setting GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    #loading network
    if opt.Network_mode != 'ST':
        net = model.subnet(in_channels=opt.in_channels, n_classes=opt.n_classes)
    else:
        net = model.MDN(in_channels=opt.in_channels, n_classes=opt.n_classes)
    #load trained model
    #restore_path = os.path.join(opt.saveroot, 'checkpoints', '26800.pth')
    bestmodelpath= os.path.join(opt.saveroot, 'best_model',natsort.natsorted(os.listdir(os.path.join(opt.saveroot, 'best_model')))[-1])
    restore_path = os.path.join(opt.saveroot, 'best_model',natsort.natsorted(os.listdir(os.path.join(opt.saveroot, 'best_model')))[-1])+'/'+os.listdir(bestmodelpath)[0]
    net.load_state_dict(
        torch.load(restore_path, map_location=device)
    )
    #input the model into GPU
    net.to(device=device)
    try:
        test_net(net=net,device=device)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
