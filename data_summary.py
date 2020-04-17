import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('ps')
# ['GTK3Agg', 'GTK3Cairo', 'MacOSX', 'nbAgg', 'Qt4Agg', 'Qt4Cairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 
# 'WXAgg', 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']

import os
from sklearn.model_selection import train_test_split
import shutil
import tqdm
from collections import Counter
import numpy as np
import cv2

def data_summary(data_dir, data_name, phase):
    print('Dataset', data_name)
    train_list = [f for f in  os.listdir(data_dir) if f.endswith('.jpg') or f.endswith('.png')]

    print('#image', len(train_list))

    img_ids = []
    person_ids = []
    person_ids = []
    for line in train_list:
        img_id, _ = os.path.splitext(line)
        img_ids.append(img_id)

        person_id = line.split('_')[0]
        person_ids.append(person_id)

    counter = Counter(person_ids)
    print('#PID', len(counter.keys()))
    #print('counter_values', counter.values())

    values = np.asarray(list(counter.values()))
    #print('values length', values.shape)

    print('min', values.min(), 'max', values.max(), 'mean', np.around(values.mean(), 2))

    #for i in range(1, 150):
    #    print('%d counts' % i, (values == i).sum())


    #plt.bar(counter.keys(), counter.values())
    plt.rcParams.update({'font.size': 18})
    plt.rcParams.update({'figure.autolayout': True})
    plt.figure()
    plt.hist(counter.values(), bins=50)
    plt.xlabel('Image Number')
    plt.ylabel('Person ID Number')
    #plt.show()
    plt.savefig('%s_%s.pdf' % (phase, data_name))
    plt.close()


def get_img_mean_std(data_dir):
    img_paths = [p for p in os.listdir(data_dir) if p.endswith('.png')]
    n = len(img_paths)

    if n > 40000:
        img_paths = np.random.choice(img_paths, size=40000, replace=False)
        n = len(img_paths)

    imgs = np.zeros((n, 256, 128, 3))
    for i, img_path in enumerate(tqdm.tqdm(img_paths)):
        #print('img_path', img_path)
        img = cv2.imread(os.path.join(data_dir, img_path))
        imgs[i] = img

    for i in range(imgs.shape[3]):
        print('%d mean, std' % i, imgs[...,i].mean() / 255.0, imgs[...,i].std()/ 255.0)


if __name__ == '__main__':
    data_dir = '../data/Market-1501-v15.09.15/bounding_box_train'
    data_summary(data_dir, 'market1501', 'train')

    data_dir = '../data/DukeMTMC-reID/bounding_box_train'
    data_summary(data_dir, 'dukeMTMC', 'train')

    data_dir = '../batch-dropblock-network/data/cuhk-detect/bounding_box_train'
    data_summary(data_dir, 'cuhk03-detect', 'train')

    data_dir = '../data/MSMT17/bounding_box_train'
    data_summary(data_dir, 'msmt17', 'train')

    ########
    data_dir = '../data/Market-1501-v15.09.15/bounding_box_test'
    data_summary(data_dir, 'market1501', 'test')

    data_dir = '../data/DukeMTMC-reID/bounding_box_test'
    data_summary(data_dir, 'dukeMTMC', 'test')

    data_dir = '../batch-dropblock-network/data/cuhk-detect/bounding_box_test'
    data_summary(data_dir, 'cuhk03-detect', 'test')

    data_dir = '../data/MSMT17/bounding_box_test'
    data_summary(data_dir, 'msmt17', 'test')



    #data_dir = '/Volumes/Data/比赛/行人重识别2019/data/复赛/mytrain'
    #get_img_mean_std(data_dir)

