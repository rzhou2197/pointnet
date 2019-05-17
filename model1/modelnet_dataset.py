import os
import os.path
import json
import numpy as np
import sys
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


class ModelNetDataset(object):
    def __init__(self, root, batch_size=16, npoints=10000, split='train', normalize=True, normal_channel=False, modelnet10=False, cache_size=15000, shuffle=False):
        self.root = root
        self.batch_size = batch_size
        self.npoints = npoints
        self.normalize = normalize
        self.split=split
        self.shuffle = shuffle
        self.normal_channel = normal_channel
        self.get_names()
        if self.shuffle:
            np.random.shuffle(self.name_list)
        self.initPos = 0

    def get_names(self):
        self.name_list = []
        if self.split=='train':
                readfile=open(self.root+'/train.txt')
        if self.split=='test':
                readfile=open(self.root+'/test.txt')
        for line in readfile:
                self.name_list += [line.rstrip()]
        self.name_list = list(self.name_list)
    def num_channel(self):
        if self.normal_channel:
            return 6
        else:
            return 3

    def next_batch(self, augment=False):
        start_idx = self.initPos
        end_idx = min(self.initPos + self.batch_size, len(self.name_list))
        this_batch_size = end_idx - start_idx
        batch_data = np.zeros((this_batch_size, self.npoints, self.num_channel()))
        attachment_data = np.zeros((this_batch_size, 128))
        startstage_data = np.zeros((this_batch_size, 128))
        orientation_data = np.zeros((this_batch_size, 128))
        surface_data = np.zeros((this_batch_size, 128))
        type_data = np.zeros((this_batch_size, 128))
        for i in range(this_batch_size):
            name = self.name_list[self.initPos]
            batch_data[i] = np.load(os.path.join(self.root, 'pointdata', name+'.npy'))[0:self.npoints, :]
            attachment_data[i] = np.load(os.path.join(self.root, 'attachment', name+'.npy'))
            startstage_data[i] = np.load(os.path.join(self.root, 'startstage', name+'.npy'))
            orientation_data[i] = np.load(os.path.join(self.root, 'orientation', name+'.npy'))
            surface_data[i] = np.load(os.path.join(self.root, 'surface', name+'.npy'))
            type_data[i] = np.load(os.path.join(self.root, 'type', name+'.npy'))
            self.initPos += 1
            if self.normalize:
                batch_data[i] = pc_normalize(batch_data[i])
        return (batch_data, attachment_data, startstage_data, orientation_data, surface_data, type_data)

    def reset(self):
        self.initPos=0

    def has_next_batch(self):
        if self.initPos >= len(self.name_list) - 1:
            if self.shuffle:
                np.random.shuffle(self.name_list)
            self.initPos = 0
            return False
        else:
            return True


if __name__ == '__main__':
    d = ModelNetDataset(root='alldata/', split='train')
    while d.has_next_batch():
        batch_data, attachment_data, startstage_data, orientation_data, surface_data, type_data = d.next_batch()
        print(batch_data.shape, startstage_data.shape, orientation_data.shape, surface_data.shape, type_data.shape)
        print(startstage_data[0])
                                                                                                                                                                                          

