from matplotlib import pyplot as plt
import json
import os.path as osp
import shutil
import tqdm
import os
import h5py
import numpy as np

def prepare_fine_tune_data(json_path, dist_path, image_folder, thresh):
    with open(json_path, 'r') as f:
        results = json.load(f)

    with h5py.File(dist_path, 'r') as f:
        dist_mat = f['dist_mat'][()]

    indicies = np.argsort(dist_mat, axis=1)


    write_dir = osp.join(image_folder, f'fine_tune_{thresh:.1f}')
    if not osp.isdir(write_dir):
        os.mkdir(write_dir)

    for idx, query_path in enumerate(tqdm.tqdm(results.keys())):
        full_query_path = osp.join(image_folder, 'query_a', query_path)
        shutil.copy(full_query_path, osp.join(image_folder, f'fine_tune_{thresh:.1f}', str(idx) + '_' + query_path))

        for i in range(200):
            if dist_mat[idx][indicies[idx, i]] < thresh:
                gallery_path = results[query_path][i]
                img_path = osp.join(image_folder, 'gallery_a', gallery_path)
                shutil.copy(img_path, osp.join(image_folder, f'fine_tune_{thresh:.1f}', str(idx) + '_' + gallery_path))
            else:
                break

        #if idx >= 10:
        #    break

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument('--json_path', type=str, help="json_path", default='submit/reid_resnext101_ibn_ft_flip_rerank_cross.json')
    parser.add_argument('--dist_path', type=str, help="dist_path", default='dist_mats/test_resnext101_ibn_ft_flip_rerank_cross.h5')
    parser.add_argument('--image_folder', type=str, help="image_folder",
                        default='../data/复赛/测试集A')
    parser.add_argument('--thresh', type=float, help="threshold")
    args = parser.parse_args()

    json_path = args.json_path
    dist_path = args.dist_path
    image_folder = args.image_folder
    thresh = args.thresh
    prepare_fine_tune_data(json_path, dist_path, image_folder, thresh)


