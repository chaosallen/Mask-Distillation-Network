import numpy as np
import Kappa

namelist=np.load('/home/limingchao/PycharmProjects/untitled/BJ_baseline/logs/results/namelist.npy')
gtlist=np.load('/home/limingchao/PycharmProjects/untitled/BJ_baseline/logs/results/gtlist.npy')
model1=np.load('/home/limingchao/PycharmProjects/untitled/BJ_baseline/logs/results/SqueezeNet_T.npy')
model2=np.load('/home/limingchao/PycharmProjects/untitled/BJ_baseline/logs/results/SENet_T.npy')
model3=np.load('/home/limingchao/PycharmProjects/untitled/BJ_baseline/logs/results/VGG16_T.npy')
model4=np.load('/home/limingchao/PycharmProjects/untitled/BJ_baseline/logs/results/VGG19_T.npy')
model5=np.load('/home/limingchao/PycharmProjects/untitled/BJ_baseline/logs/results/ResNet101_T.npy')
model6=np.load('/home/limingchao/PycharmProjects/untitled/BJ_baseline/logs/results/Xception_T.npy')
model7=np.load('/home/limingchao/PycharmProjects/untitled/BJ_baseline/logs/results/ResNet50_T.npy')
model8=np.load('/home/limingchao/PycharmProjects/untitled/BJ_baseline/logs/results/DenseNet169_T.npy')
num=namelist.size
studentv4=np.zeros(num)
studentv6=np.zeros(num)
studentv8=np.zeros(num)

for i,name in enumerate(namelist):
    studentv4[i]=np.argmax(np.bincount([model1[i],model2[i],model3[i],model4[i]]))
    studentv8[i] = np.argmax(np.bincount([model1[i], model2[i], model3[i], model4[i], model5[i], model6[i],model7[i], model8[i]]))
kappa_s4 = Kappa.quadratic_weighted_kappa(studentv4, gtlist)
kappa_s8 = Kappa.quadratic_weighted_kappa(studentv8, gtlist)
acc_s4=Kappa.acc(studentv4, gtlist)
acc_s8=Kappa.acc(studentv8, gtlist)
print(acc_s4,acc_s8)
print(kappa_s4,kappa_s8)





