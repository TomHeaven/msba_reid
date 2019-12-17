# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re

import os.path as osp

from .bases import ImageDataset


class Competition1910(ImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = '复赛'
    test_data_dir = '复赛/测试集A'

    def __init__(self, root, test_phase=False, **kwargs):
        #super(Competition1910, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        if not test_phase:
            self.train_dir = osp.join(self.dataset_dir, 'mytrain')
            self.query_dir = osp.join(self.dataset_dir, 'myval_query')
            self.gallery_dir = osp.join(self.dataset_dir, 'myval_gallery')
        else:
            #self.train_dir = osp.join(self.dataset_dir, 'mytrain')
            self.train_dir = osp.join(root, self.test_data_dir, 'myquery') # not used
            self.query_dir = osp.join(root, self.test_data_dir, 'myquery')
            self.gallery_dir = osp.join(root, self.test_data_dir, 'mygallery')

        self._check_before_run()

        if not test_phase:
            train = self._process_dir(self.train_dir, relabel=True)
        else:
            ## place holder, not used
            train = self._process_dir(self.query_dir, relabel=False)

        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        # DEBUG
        #query = query[:2]
        super(Competition1910, self).__init__(train, query, gallery, **kwargs)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.png'))
        #print('dir_path', dir_path, 'img_paths', len(img_paths), img_paths[0])
        #pattern = re.compile(r'([-\d]+)_c(\d)')

        DEBUG = False
        if DEBUG:
            img_paths = img_paths[:200]

        pid_container = set()
        for img_path in img_paths:
            #print('img_path', img_path.split('/')[-1])
            #pid, _ = map(int, pattern.search(img_path.split('/')[-1]).groups())
            pid = int(img_path.split('/')[-1].split('_')[0])
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            #pid, camid = map(int, pattern.search(img_path).groups())
            pid = int(img_path.split('/')[-1].split('_')[0])
            camid = int(img_path.split('/')[-1].split('_')[1][:-4])

            if pid == -1:
                continue  # junk images are just ignored
            #print('pid', pid)
            #assert 0 <= pid <= 4768  # pid == 0 means background
            #assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset
