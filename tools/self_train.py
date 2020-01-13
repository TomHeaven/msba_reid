from __future__ import print_function, absolute_import
import argparse
import os 
import numpy as np
from sklearn.cluster import DBSCAN

import sys
sys.path.append('.')
from config import cfg
from utils.logger import setup_logger
from modeling import build_model
from data import get_dataloader
#from utils.dist_metric import DistanceMetric
from torch.backends import cudnn
from engine.inference import inference_ssg
from engine.trainer import ReidSystem

import torch

def compute_distmat_using_gpu(probFea, galFea, distmat=None, theta=1.0, memory_save=True, mini_batch=5000):
    """

    :param probFea:
    :param galFea:
    :param dist_mat: local_distmat, re-use the variable to save memory
    :param theta:
    :param memory_save:
    :param mini_batch:
    :return:
    """
    print('Computing distance using GPU ...')
    if memory_save:
        if distmat is None:
            #print('distmat is None')
            distmat = np.zeros((probFea.size(0), galFea.size(0)), dtype=np.float16)

        i = 0
        while True:
            it = i + mini_batch
            # print('i, it', i, it)
            if it < probFea.size()[0]:
                distmat[i:it, :] = torch.pow(torch.cdist(probFea[i:it, :], galFea), 2).cpu().numpy() * theta + (1 - theta) * distmat[i:it, :]
            else:
                distmat[i:, :] = torch.pow(torch.cdist(probFea[i:, :], galFea), 2).cpu().numpy() * theta + (1 - theta) * distmat[i:, :]
                break
            i = it
    else:
        ### new API
        if distmat is None:
            distmat = torch.pow(torch.cdist(probFea, galFea), 2)
        else:
            distmat = torch.pow(torch.cdist(probFea, galFea), 2) * theta + (1 - theta) * distmat

    return distmat



def main():
    parser = argparse.ArgumentParser(description="ReID Model Training")
    parser.add_argument(
        '-cfg', "--config_file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str
    )
    # parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    cfg.freeze()

    log_save_dir = os.path.join(cfg.OUTPUT_DIR, cfg.DATASETS.TEST_NAMES, cfg.MODEL.VERSION)
    if not os.path.exists(log_save_dir): os.makedirs(log_save_dir)

    logger = setup_logger("reid_baseline.train", log_save_dir, 0)
    logger.info("Using {} GPUs.".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    logger.info('start training')
    cudnn.benchmark = True

    # Create data loaders
    tng_dataloader, val_dataloader, num_classes, num_query, tng_set, val_set = get_dataloader(cfg, return_sets=True)

    # Start training
    writer = None
    reid_system = ReidSystem(cfg, logger, writer)
    model = reid_system.model
    model.cuda()

    iter_nums = 1
    cluster_list = []
    #top_percent = args.rho
    for iter_n in range(0, iter_nums):
        #### get source datas' feature
        source_features = inference_ssg(cfg, model, tng_dataloader)
        target_features = inference_ssg(cfg, model, val_dataloader)

        #### calculate distance and rerank result
        distmat = compute_distmat_using_gpu(source_features, target_features)
        print('source_features', source_features.shape, 'target_features', target_features.shape, 'distmat', distmat.shape)
        
        labels_list, cluster_list = generate_self_label(distmat, iter_n, cluster_list)
        #### generate new dataset
        val_set = update_dataset(val_set, labels_list)
        del labels_list
        # del cluster_list
        reid_system.train()

def generate_self_label(e_dist, n_iter, cluster_list=[]):
    labels_list = []
    for s in range(len(e_dist)):
        if n_iter==0:
            tmp_dist = e_dist[s]
            # DBSCAN cluster
            tri_mat = np.triu(tmp_dist, 1)       # tri_mat.dim=2
            tri_mat = tri_mat[np.nonzero(tri_mat)] # tri_mat.dim=1
            tri_mat = np.sort(tri_mat, axis=None)
            top_num = np.round(cfg.SOLVER.RHO*tri_mat.size).astype(int)
            eps = tri_mat[:top_num].mean()
            print('eps in cluster: {:.3f}'.format(eps))
            cluster = DBSCAN(eps=eps,min_samples=4, metric='precomputed', n_jobs=8)
            cluster_list.append(cluster)
        else:
            cluster = cluster_list[s]
        #### select & cluster images as training set of this epochs
        print('Clustering and labeling...')
        labels = cluster.fit_predict(e_dist[s])
        num_ids = len(set(labels)) - 1  ##for DBSCAN cluster
        print('Iteration {} have {} training ids'.format(n_iter+1, num_ids))
        labels_list.append(labels)
        del labels
        del cluster
    return labels_list, cluster_list

def update_dataset(tgt_dataset, labels_list):
    new_dataset = []
    for i, (img_path, pid, camid) in enumerate(tgt_dataset.train):
        label = []
        for s in range(len(labels_list)):
            label.append(labels_list[s][i])
        if -1 in label:
            continue
        new_dataset.append((img_path, label, camid))

    tgt_dataset.train = new_dataset
    return tgt_dataset

if __name__ == '__main__':
    main()
