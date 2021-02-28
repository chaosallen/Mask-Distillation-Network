import numpy as np
import os
import random
import scipy.misc as misc
import torch
import imageio
import matplotlib.pyplot as plt

class Batchreader:

    def __init__(self, path):
        print("Initializing Batch Dataset Reader...")
        self.namelist = {};
        self.path = path
        self.class_list = os.listdir(os.path.join(path,'Train','S'))
        for classname in self.class_list:
            self.namelist[classname]=os.listdir(os.path.join(path,'Train','S',classname))

    def CBM(self,image,r,g,b):
        image[0, :, :] = image[0, :, :] * r
        image[1, :, :] = image[1, :, :] * g
        image[2, :, :] = image[2, :, :] * b
        return image

    def next_batch(self, batch_size,input_size):
        images_student = np.arange(batch_size*3* input_size[0]*input_size[1]).reshape(batch_size,3,input_size[0],input_size[1])
        images_teacher = np.arange(batch_size*3* input_size[0]*input_size[1]).reshape(batch_size,3,input_size[0],input_size[1])
        annotations = np.arange(batch_size).reshape(batch_size)
        for batch in range(batch_size) :
            classname = batch % 4+1
            id=random.randint(0,len(self.namelist[str(classname)])-1)
            name=self.namelist[str(classname)][id]
            path_student=os.path.join(self.path,'Train','S',str(classname), name)
            path_teacher= os.path.join(self.path, 'Train','T', str(classname), name)

            p = 0.005
            r = random.gauss(1, p)
            g = random.gauss(1, p)
            b = random.gauss(1, p)
            images_student[batch, :, :, :] = self.CBM(np.array(imageio.imread(path_student)).transpose(2, 0, 1),r,g,b)
            images_teacher[batch, :, :, :] = self.CBM(np.array(imageio.imread(path_teacher)).transpose(2, 0, 1),r,g,b)
            #images_student[batch, :, :, :] = misc.imresize(np.array(imageio.imread(path_student)), [input_size[0], input_size[1]],interp='nearest').transpose(2, 0, 1)
            #images_teacher[batch, :, :, :] = misc.imresize(np.array(imageio.imread(path_teacher)), [input_size[0], input_size[1]],interp='nearest').transpose(2, 0, 1)

            switch=random.randint(0,2)
            if switch==0:
                images_student[batch, :, :, :] = images_student[batch,:,:,::-1]
                images_teacher[batch, :, :, :] = images_teacher[batch, :, :, ::-1]
            annotations[batch] = np.array(int(classname)-1)
        images_student = torch.from_numpy(images_student)
        images_teacher = torch.from_numpy(images_teacher)
        annotations = torch.from_numpy(annotations)
        return images_student,images_teacher, annotations


