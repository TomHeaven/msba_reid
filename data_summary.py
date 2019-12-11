import os
from sklearn.model_selection import train_test_split
import shutil
import tqdm
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import cv2

def data_summary(data_dir, list_file):
    list_file = os.path.join(data_dir, list_file)
    with open(list_file, 'r') as f:
        train_list = f.readlines()
        #print('train_list', train_list)

    img_ids = []
    person_ids = []
    for line in train_list:
        img_id, _ = os.path.splitext(line.split('/')[1].split(' ')[0])
        img_ids.append(img_id)

        person_id = line.split(' ')[1][:-1]
        person_ids.append(person_id)


    train_person_ids, val_person_ids = train_test_split(person_ids, test_size=0.1, random_state=20191029)

    counter = Counter(person_ids)
    print('counter_keys', counter.keys())
    print('counter_values', counter.values())

    values = np.asarray(list(counter.values()))
    print('values length', values.shape)

    print('min', values.min(), 'max', values.max(), 'mean', np.around(values.mean(), 2))

    for i in range(1, 100):
        print('%d counts' % i, (values == i).sum())


    #plt.bar(counter.keys(), counter.values())
    #plt.xlabel('person id')
    #plt.ylabel('number')
    #plt.show()


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
    data_dir = '/Volumes/Data/比赛/行人重识别2019/data/复赛'
    list_file = 'train_list.txt'
    #data_summary(data_dir, list_file)

    data_dir = '/Volumes/Data/比赛/行人重识别2019/data/复赛/mytrain'
    get_img_mean_std(data_dir)

