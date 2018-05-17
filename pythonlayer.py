import sys
import os
import cv2
import caffe
import numpy as np
import random
import torch
import pandas as pd
#from torch.utils.serialization import load_lua
#import py_utils
from multiprocessing import Queue 
from threading import Thread


################################################################################
#########################Data Layer By Python###################################
################################################################################
class Data_Layer_train(caffe.Layer):
    def setup(self, bottom, top):
       
        if self.phase == caffe.TRAIN :
            print '~~~~~~~~~~~~~~~train'
            self.batch_size = 1
            self.img_file = '/data_1/data/formatted_trainval/shanghaitech_part_A_patches_2/train.txt'
            self.img_pieces_path = '/data_1/data/formatted_trainval/split_2/'
            self.batch_loader = BatchLoader(self.img_file,self.img_pieces_path, 'train')
        else:
            print '~~~~~~~~~~~~~~~test'
            self.batch_size = 1
            self.img_file = '/data_1/data/formatted_trainval/shanghaitech_part_A_patches_2/val.txt'
            self.img_pieces_path = '/data_1/data/formatted_trainval/val_split_2/'
            self.batch_loader = BatchLoader(self.img_file,self.img_pieces_path, 'test')
        #img_root = '/media/pengshanzhen/bb42233c-19d1-4423-b161-e5256766be8e/300/300W_LP'
        #land_mark_root = '/media/pengshanzhen/bb42233c-19d1-4423-b161-e5256766be8e/300/landmarks'
        #self.batch_loader = BatchLoader(img_root, land_mark_root, self.img_w)
        
    def reshape(self, bottom, top):
        self.img, self.den,self.im_toget = self.batch_loader.load_next_image()

        top[0].reshape(self.img.shape[0], self.img.shape[1], self.img.shape[2],
                    self.img.shape[3])
        top[1].reshape(self.den.shape[0], self.den.shape[1],
                    self.den.shape[2], self.den.shape[3])
        top[2].reshape(self.im_toget.shape[0], self.im_toget.shape[1],
                    self.im_toget.shape[2], self.im_toget.shape[3])

    def forward(self, bottom, top):
        
        top[0].data[ ...] = self.img
        top[1].data[ ...] = self.den
        top[2].data[ ...] = self.im_toget
        
        # img=top[0].data[0]
        # img=np.transpose(img, (1,2,0))
        # print img[128,128,0]
        # img=img*128+127.5
        # print img[128,128,0]
        # print(img.shape)
        # cv2.imshow("as", img)
        # cv2.waitKey(1)

    def backward(self, top, propagate_down, bottom):
        pass

class BatchLoader(object):
    def __init__(self, img_file,img_pieces_path, phase):
        self.pts_list = []
        self.phase = phase
        self.mean = 127.5
        with open(img_file)as f:
            img_lines = f.readlines()
        for img_line in img_lines:
            index = img_line.strip().rfind(' ')
            index1 = img_line.strip().rfind('/n')
            img_path = img_line[:index]
            label_path = img_line[index+1:index1]
            img_fname = img_path.split('/')[6]
            img_main = img_fname.split('.')[0]

            img_pieces_1_path = img_pieces_path +img_main +'_1'+'_1'+'.jpg'
            img_pieces_2_path = img_pieces_path +img_main +'_1'+'_2'+'.jpg'
            img_pieces_3_path = img_pieces_path +img_main +'_2'+'_1'+'.jpg'
            img_pieces_4_path = img_pieces_path +img_main +'_2'+'_2'+'.jpg'
            #den = pd.read_csv(label_path,sep=',',header=None).as_matrix()
            #den  = den.astype(np.float32, copy=False)
            #count = np.sum(den)

        
            if not os.path.isfile(img_path):
                continue
            #if count == 0:
            #    continue
               
            self.pts_list.append([img_path, label_path,img_pieces_1_path,img_pieces_2_path,img_pieces_3_path,img_pieces_4_path])
        random.shuffle(self.pts_list)
        self.pts_cur = 0
        self.record_queue = Queue(maxsize=128)
        self.image_label_queue = Queue(maxsize=128)
        self.thread_num = 6
        t_record_producer = Thread(target=self.record_producer)
        t_record_producer.daemon = True 
        t_record_producer.start()
        for i in range(self.thread_num):
            t = Thread(target=self.record_customer)
            t.daemon = True
            t.start() 
        print 'load data done....'
        
    def load_next_image(self): 
        img, den,im_toget = self.image_label_queue.get()
        return img, den,im_toget

    def record_producer(self):
        while True:
            if self.pts_cur == len(self.pts_list):
                self.pts_cur = 0
                random.shuffle(self.pts_list)
            self.record_queue.put(self.pts_list[self.pts_cur])
            self.pts_cur += 1

    def record_process(self, cur_data):
        image_file_name = cur_data[0]
        img = cv2.imread(image_file_name,0)
        img = img.astype(np.float32, copy=False)
        ht = img.shape[0]
        wd = img.shape[1]
        ht_1 = (ht/8)*8
        wd_1 = (wd/8)*8
        img = cv2.resize(img,(wd_1,ht_1))
        img = img.reshape((1,1,img.shape[0],img.shape[1]))
        #image = cv2.imread(image_file_name)
        
        
        
        label_path = cur_data[1]
        den = pd.read_csv(label_path,sep=',',header=None).as_matrix()
        den  = den.astype(np.float32, copy=False)
        wd_1 = wd_1/8
        ht_1 = ht_1/8
        den = cv2.resize(den,(wd_1,ht_1))                
        den = den * ((wd*ht)/(wd_1*ht_1))
        den = den.reshape((1,1,den.shape[0],den.shape[1]))

        image_1_path = cur_data[2]
        im_1=cv2.imread(image_1_path,0)
        im_1 = cv2.resize(im_1,(128,128))
        im_1 = (im_1 -127.5)*0.0078125
        im_1 = im_1.reshape((1,128,128))
        image_2_path = cur_data[3]
        im_2=cv2.imread(image_2_path,0)
        im_2 = cv2.resize(im_2,(128,128))
        im_2 = (im_2 -127.5)*0.0078125
        im_2 = im_2.reshape((1,128,128))
        image_3_path = cur_data[4]
        im_3 = cv2.imread(image_3_path,0)
        im_3 = cv2.resize(im_3,(128,128))
        im_3 = (im_3 -127.5)*0.0078125
        im_3 = im_3.reshape((1,128,128))
        image_4_path = cur_data[5]
        im_4 = cv2.imread(image_4_path,0)
        im_4 = cv2.resize(im_4,(128,128))
        im_4 = (im_4 -127.5)*0.0078125
        im_4 = im_4.reshape((1,128,128))

        im_toget = np.zeros((4,1,128,128))
        im_toget[0] = im_1
        im_toget[1] = im_2
        im_toget[2] = im_3
        im_toget[3] = im_4  

      
        #label = label.reshape(136)
        return [img, den,im_toget]

    def record_customer(self):
        while True:
            item = self.record_queue.get()
            out = self.record_process(item)
            self.image_label_queue.put(out)


#####################################################################################
class score_to_featuremaps_Layer(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("Need 2 Inputs")
    def reshape(self, bottom, top):
        top[0].reshape(1, 5,(bottom[1].data.shape[2]),(bottom[1].data.shape[3]))

    def forward(self, bottom, top):
        score_density_map = np.zeros((5,(bottom[1].data.shape[2]),(bottom[1].data.shape[3])))
        for j in range(5):
          score_density_map[j][0:(bottom[1].data.shape[2])/2,0:(bottom[1].data.shape[3])/2] =bottom[0].data[0][j]
        for k in range(5):
          score_density_map[k][(bottom[1].data.shape[2])/2:(bottom[1].data.shape[2]),0:(bottom[1].data.shape[3])/2] =bottom[0].data[1][k]
        for l in range(5):
          score_density_map[l][0:(bottom[1].data.shape[2])/2,(bottom[1].data.shape[3])/2:(bottom[1].data.shape[3])] =bottom[0].data[2][l]
        for m in range(5):
          score_density_map[m][(bottom[1].data.shape[2])/2:(bottom[1].data.shape[2]),(bottom[1].data.shape[3])/2:(bottom[1].data.shape[3])] =bottom[0].data[3][m]
        score_density_map = score_density_map.reshape((1,5,(bottom[1].data.shape[2]),(bottom[1].data.shape[3])))
        top[0].data[...] = score_density_map
        
    def backward(self, top, propagate_down, bottom):
        for i in range(2):
            bottom[i].diff[...][...] = 0

















