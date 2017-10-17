from __future__ import print_function

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="2"


import csv
import os
import os.path
import shutil
import time
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pickle

import VGG_FACE
import threading


def img_loader(path):
    return Image.open(path).convert('RGB')
  
def outpath(path, output_path):

    dir_path =  os.path.dirname(path)
    image_name = os.path.basename(path)
    last_dir_path = dir_path.split('/')[-1]
    return output_path + '/' + last_dir_path + '/' + image_name
  
def write_data(data, path):
    dir_path =  os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(path, 'wb') as stream:
        pickle.dump(data, stream)
    print("Saving data to",path)
    
    
class ThreadingExample(object):
    """ Threading example class
    The run() method will be started and it will run in the background
    until the application exits.
    """

    def __init__(self, model, images, paths):
        """ Constructor
        :type interval: int
        :param interval: Check interval, in seconds
        """
        self.model = model
        self.images = images
        self.paths = paths

        self.thread = threading.Thread(target=self.run, args=())
        self.thread.start()                                  # Start the execution

    def run(self):
        """ Method that runs forever """
        self.images = torch.stack(self.images, dim=0)        
        image_var = torch.autograd.Variable(self.images, volatile=True)
        data = self.model(image_var)
        data = data.data.cpu().numpy()
        for i, p in enumerate(self.paths):
            write_data(data[i], p)
        

def main():

    print("Loading model...")
    list_of_images = sys.argv[1]
    output_path = sys.argv[2]
    model = VGG_FACE.VGG_FACE
    print("...")
    model.load_state_dict(torch.load('VGG_FACE.pth'))
    print("...")
    for param in model.parameters():
        param.requires_grad = False
    list_model = list(model.children())
    del list_model[-1] #delete softmax
    del list_model[-1] #delete last dense layer
    #list_model[-1] =  torch.nn.Sequential(VGG_FACE.Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),torch.nn.Linear(4096,64))
    #list_model.append(  nn.ReLU() )
    #list_model.append(  nn.Dropout(0.5) )
    #list_model.append( torch.nn.Sequential(VGG_FACE.Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),torch.nn.Linear(64,7)) )
    print("...")
    model =  nn.Sequential(*list_model)
    model = torch.nn.DataParallel(model).cuda()

    model.eval()
    print("Model Loaded.")
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    scale = transforms.Scale(224)
    to_tensor = transforms.ToTensor()

    back_process = None
    
    with open(list_of_images,'r') as list_stream:
        images = []
        paths = []
        for f in list_stream:
            f = f.strip()
            #print(f)
            image = img_loader(f)
            image = scale(image)
            image = to_tensor(image)
            image = normalize(image)
            #print(image.size())
            
            images.append(image)
            paths.append(outpath(f, output_path))
            if len(images) == 24:
                
                
                if True:
                    if back_process is not None:
                        back_process.thread.join()
                        back_process = None
                    
                    back_process = ThreadingExample( model, images.copy(), paths.copy() )
                
                else:
                  
                    images = torch.stack(images, dim=0)
                    #print(images.size())
                    #print(paths)
                    #do things
                    image_var = torch.autograd.Variable(images, volatile=True)
                    data = model(image_var)
                    data = data.data.cpu().numpy()
                    for i, d in enumerate(paths):
                        #print(paths[i])
                        #print(data[i].shape)
                        write_data(data[i], paths[i])
                images = []
                paths = []
                
                #break
        if len(images) != 0:
            images = torch.stack(images, dim=0)
            #print(images.size())
            #print(paths)
            #do things
            image_var = torch.autograd.Variable(images, volatile=True).cuda()
            data = model(image_var)
            data = data.data.cpu().numpy()
            for i, d in enumerate(paths):
                #print(paths[i])
                #print(data[i].shape)
                write_data(data[i], paths[i])
            images = []
            paths = []
        
    


if __name__ == '__main__':
    main()
