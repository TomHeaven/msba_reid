from matplotlib import pyplot as plt
import json
import os.path as osp
import shutil
import tqdm
import os

def visualize(json_path, image_folder, top_k):
    with open(json_path, 'r') as f:
        results = json.load(f)

    write_dir = osp.join(image_folder, f'fine_tune_{top_k}')
    if not osp.isdir(write_dir):
        os.mkdir(write_dir)

    for idx, query_path in enumerate(tqdm.tqdm(results.keys())):
        full_query_path = osp.join(image_folder, 'query_a', query_path)
        shutil.copy(full_query_path, osp.join(image_folder, f'fine_tune_{top_k}', str(idx) + '_' + query_path))

        for i in range(top_k):
            gallery_path = results[query_path][i]
            img_path = osp.join(image_folder, 'gallery_a', gallery_path)
            #print(img_path)
            shutil.copy(img_path, osp.join(image_folder, f'fine_tune_{top_k}', str(idx) + '_' + gallery_path))

        #if idx >= 10:
        #    break

if __name__ == '__main__':
    json_path = 'submit/reid_resnet101_ibn_20191229_144223_flip_rerank_cross_77.7.json'
    image_folder = '../data/复赛/测试集A'
    top_k = 1
    visualize(json_path, image_folder, top_k)


