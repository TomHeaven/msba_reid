import os
from sklearn.model_selection import train_test_split
import shutil
import tqdm

def preprocess_train(data_dir, list_file):
    list_file = os.path.join(data_dir, list_file)
    with open(list_file, 'r') as f:
        train_list = f.readlines()
        #print('train_list', train_list)

    img_id_set = set()
    person_id_set = set()
    for line in train_list:
        img_id, _ = os.path.splitext(line.split('/')[1].split(' ')[0])
        img_id_set.add(img_id)

        person_id = line.split(' ')[1][:-1]
        person_id_set.add(person_id)

    img_ids = list(img_id_set)
    person_ids = list(person_id_set)
    print('img_ids', len(img_ids), img_ids)
    print('person_ids', len(person_ids), person_ids)

    train_person_ids, val_person_ids = train_test_split(person_ids, test_size=0.1, random_state=20191029)

    for line in tqdm.tqdm(train_list):
        img_path = line.split('/')[1].split(' ')[0]
        img_id, img_ext = os.path.splitext(line.split('/')[1].split(' ')[0])
        person_id = line.split(' ')[1][:-1]

        out_image_name = person_id + '_' +  img_id + img_ext
        #print('img_path', img_path, 'out_image_name', out_image_name)

        if person_id in train_person_ids:
            folder = 'mytrain'
        elif person_id in val_person_ids:
            folder = 'myval'
        else:
            raise Exception("Unknown person id", person_id)

        out_image_path = os.path.join(data_dir, folder, out_image_name)
        if not os.path.isdir(os.path.join(data_dir, folder)):
            os.mkdir(os.path.join(data_dir, folder))

        shutil.copy(os.path.join(data_dir, 'train_set', img_path), out_image_path)



def preprocess_query(data_dir):
    img_files = [fname for fname in os.listdir(os.path.join(data_dir, 'query_b')) if fname.endswith('.png')]

    #if not os.path.isdir(os.path.join(data_dir, 'myquery')):
    #    os.mkdir(os.path.join(data_dir, 'myquery'))

    for person_id, fname in enumerate(tqdm.tqdm(img_files)):
        img_id, img_ext = os.path.splitext(fname)

        out_image_name = str(person_id) + '_' + img_id + img_ext
        # print('img_path', img_path, 'out_image_name', out_image_name)
        folder = 'myquery'

        out_image_path = os.path.join(data_dir, folder, out_image_name)
        if not os.path.isdir(os.path.join(data_dir, folder)):
            os.mkdir(os.path.join(data_dir, folder))
        shutil.copy(os.path.join(data_dir, 'query_b', fname), out_image_path)


def preprocess_gallery(data_dir):
    img_files = [fname for fname in os.listdir(os.path.join(data_dir, 'gallery_b')) if fname.endswith('.png')]

    if not os.path.isdir(os.path.join(data_dir, 'mygallery_b')):
        os.mkdir(os.path.join(data_dir, 'mygallery_b'))

    for person_id, fname in enumerate(tqdm.tqdm(img_files)):
        img_id, img_ext = os.path.splitext(fname)

        out_image_name = str(person_id) + '_' +  img_id + img_ext
        #print('img_path', img_path, 'out_image_name', out_image_name)
        folder = 'mygallery_b'

        out_image_path = os.path.join(data_dir, folder, out_image_name)
        if not os.path.isdir(os.path.join(data_dir, folder)):
            os.mkdir(os.path.join(data_dir, folder))

        shutil.copy(os.path.join(data_dir, 'gallery_b', fname), out_image_path)


def split_myval(data_dir):
    img_files = [fname for fname in os.listdir(os.path.join(data_dir, 'myval')) if fname.endswith('.png')]

    img_id_set = set()
    person_id_set = set()

    person_img_dict = {}
    for line in img_files:
        #print('line', line)
        img_id, _ = os.path.splitext(line.split('_')[1])
        img_id_set.add(img_id)

        person_id = line.split('_')[0]
        person_id_set.add(person_id)

        if not person_id in person_img_dict.keys():
            person_img_dict[person_id] = [img_id]
        else:
            person_img_dict[person_id].append(img_id)

    img_ids = list(img_id_set)
    person_ids = list(person_id_set)
    print('img_ids', len(img_ids), img_ids)
    print('person_ids', len(person_ids), person_ids)


    if not os.path.isdir(os.path.join(data_dir, 'myval_query')):
        os.mkdir(os.path.join(data_dir, 'myval_query'))

    if not os.path.isdir(os.path.join(data_dir, 'myval_gallery')):
        os.mkdir(os.path.join(data_dir, 'myval_gallery'))

    query = []
    gallery = []
    for k in person_img_dict.keys():
        v = person_img_dict[k]
        if len(v) > 6:
            fname = k + '_' + v[0] + '.png'
            query.append(fname)

            shutil.copy(os.path.join(data_dir, 'myval', fname), os.path.join(data_dir, 'myval_query', fname))

            for i in range(1, len(v)):
                fname = k + '_' + v[i] + '.png'
                gallery.append(fname)
                shutil.copy(os.path.join(data_dir, 'myval', fname), os.path.join(data_dir, 'myval_gallery', fname))

    print('query', len(query), query)
    print('gallery', len(gallery), gallery)

if __name__ == '__main__':
    data_dir = '/Volumes/Data/比赛/行人重识别2019/data/初赛训练集'
    list_file = 'train_list.txt'
    #preprocess_train(data_dir, list_file)
    #split_myval(data_dir)

    data_dir = '/Volumes/Data/比赛/行人重识别2019/data/初赛B榜测试集'
    preprocess_query(data_dir)


    #data_dir = '/Volumes/Data/比赛/行人重识别2019/data/初赛A榜测试集'
    data_dir = '/Volumes/Data/比赛/行人重识别2019/data/初赛B榜测试集'
    #preprocess_gallery(data_dir)

