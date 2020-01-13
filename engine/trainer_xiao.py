# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import os
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from data import get_dataloader
from data.datasets.eval_reid import evaluate
from data.prefetcher import data_prefetcher
from modeling_xiao import build_model
from modeling_xiao.baseline_parts import *
from modeling_xiao.losses import TripletLoss, CrossEntropyLabelSmooth
from solver.build import make_lr_scheduler, make_optimizer
from utils.meters import AverageMeter


class ReidSystem():
    def __init__(self, cfg, logger, writer):
        self.cfg, self.logger, self.writer = cfg, logger, writer
        # Define dataloader
        self.tng_dataloader, self.val_dataloader, self.num_classes, self.num_query = get_dataloader(cfg)
        logger.info('num_classes ' + str(self.num_classes))
        # networks
        self.model = build_model(cfg, self.num_classes)
        #self.base_type = self.model.base_type
        # loss function

        #if cfg.SOLVER.LABEL_SMOOTH:
        self.ce_loss = CrossEntropyLabelSmooth(self.num_classes)
        #else:
        #    self.ce_loss = nn.CrossEntropyLoss()
        self.triplet = TripletLoss(cfg.SOLVER.MARGIN)


        # optimizer and scheduler
        self.opt = make_optimizer(self.cfg, self.model)
        self.lr_sched = make_lr_scheduler(self.cfg, self.opt)

        self._construct()

    def _construct(self):
        self.global_step = 0
        self.current_epoch = 0
        self.batch_nb = 0
        self.max_epochs = self.cfg.SOLVER.MAX_EPOCHS
        self.log_interval = self.cfg.SOLVER.LOG_INTERVAL
        self.eval_period = self.cfg.SOLVER.EVAL_PERIOD
        self.use_dp = False
        self.use_ddp = False

    def loss_fns(self, outputs, labels):
        if len(outputs) == 2:
            triplet_loss = self.triplet(outputs[1], labels) * 0.4
            ce_loss = self.ce_loss(outputs[0], labels)
            return {'ce_loss': ce_loss, 'global_triplet_loss': triplet_loss}
        else:
            triplet_loss = self.triplet(outputs, labels)
            return {'global_triplet_loss': triplet_loss}

    def on_train_begin(self):
        self.start_epoch = 0
        self.best_mAP = -np.inf
        self.running_loss = AverageMeter()
        self.running_CE_loss = AverageMeter()
        self.running_GT_loss = AverageMeter()
        self.running_LT_loss = AverageMeter()
        self.running_OF_loss = AverageMeter()
        log_save_dir = os.path.join(self.cfg.OUTPUT_DIR, self.cfg.DATASETS.TEST_NAMES, self.cfg.MODEL.VERSION)
        self.model_save_dir = os.path.join(log_save_dir, 'ckpts')
        if not os.path.exists(self.model_save_dir): os.makedirs(self.model_save_dir)

        #######
        # Load checkpoints
        cfg = self.cfg
        if cfg.MODEL.CHECKPOINT is not '':
            self.load_checkpoint(cfg.MODEL.CHECKPOINT, with_optimizer=not cfg.MODEL.FINE_TUNE)
            #self.logger.info('continue training')
        ######

        self.gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        self.use_dp = (len(self.gpus) > 0) and (self.cfg.MODEL.DIST_BACKEND == 'dp')

        if self.use_dp:
            self.model = nn.DataParallel(self.model)

        self.model = self.model.cuda()

        self.model.train()

    def on_epoch_begin(self):
        self.batch_nb = 0
        self.current_epoch += 1
        self.t0 = time.time()
        self.running_loss.reset()
        self.running_CE_loss.reset()
        self.running_GT_loss.reset()
        self.running_LT_loss.reset()
        self.running_OF_loss.reset()

        self.tng_prefetcher = data_prefetcher(self.tng_dataloader, self.cfg)

    def training_step(self, batch):
        inputs, labels, _ = batch
        outputs = self.model(inputs, labels)

        loss_dict = self.loss_fns(outputs, labels)

        total_loss = 0
        print_str = f'\r Epoch {self.current_epoch} Iter {self.batch_nb}/{len(self.tng_dataloader)} '
        for loss_name, loss_value in loss_dict.items():
            total_loss += loss_value
            print_str += (loss_name+f': {loss_value.item():.3f} ')
        loss_dict['total_loss'] = total_loss.item()
        print_str += f'Total loss: {total_loss.item():.3f} '
        print(print_str, end=' ')
        
        if self.writer and (self.global_step+1) % self.log_interval == 0:
            if 'ce_loss' in loss_dict.keys():
                self.writer.add_scalar('cross_entropy_loss', loss_dict['ce_loss'], self.global_step)
            self.writer.add_scalar('global_triplet_loss', loss_dict['global_triplet_loss'], self.global_step)
            if 'local_triplet_loss' in loss_dict.keys():
                self.writer.add_scalar('local_triplet_loss', loss_dict['local_triplet_loss'], self.global_step)
            self.writer.add_scalar('total_loss', loss_dict['total_loss'], self.global_step)

        self.running_loss.update(total_loss.item())
        if 'ce_loss' in loss_dict.keys():
            self.running_CE_loss.update(loss_dict['ce_loss'])
        self.running_GT_loss.update(loss_dict['global_triplet_loss'])
        if 'local_triplet_loss' in loss_dict.keys():
            self.running_LT_loss.update(loss_dict['local_triplet_loss'])
        if 'of_loss' in loss_dict.keys():
            self.running_OF_loss.update(loss_dict['of_loss'])

        self.opt.zero_grad()
        total_loss.backward()
        self.opt.step()

        self.global_step += 1
        self.batch_nb += 1

    def on_epoch_end(self):
        elapsed = time.time() - self.t0
        mins = int(elapsed) // 60
        seconds = int(elapsed - mins * 60)
        print('')
        self.logger.info(f'Epoch {self.current_epoch} Total loss: {self.running_loss.avg:.3f} CE loss: {self.running_CE_loss.avg:.3f} '
                         f'GT loss: {self.running_GT_loss.avg:.3f} LT loss: {self.running_LT_loss.avg:.3f} OF loss: {self.running_OF_loss.avg:.3f} '
                         f'lr: {self.opt.param_groups[0]["lr"]:.2e} During {mins:d}min:{seconds:d}s')
        # update learning rate
        self.lr_sched.step()

    def test(self):
        # convert to eval mode
        self.model.eval()

        feats,pids,camids = [],[],[]
        val_prefetcher = data_prefetcher(self.val_dataloader, self.cfg)
        batch = val_prefetcher.next()
        while batch[0] is not None:
            img, pid, camid = batch
            with torch.no_grad():
                feat = self.model(img)
            if isinstance(feat, tuple):
                feats.append(feat[0])
            else:
                feats.append(feat)

            pids.extend(pid.cpu().numpy())
            camids.extend(np.asarray(camid))

            batch = val_prefetcher.next()

        ####
        feats = torch.cat(feats, dim=0)
        if self.cfg.TEST.NORM:
            feats = F.normalize(feats, p=2, dim=1)

        # query
        qf = feats[:self.num_query]


        q_pids = np.asarray(pids[:self.num_query])
        q_camids = np.asarray(camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]

        g_pids = np.asarray(pids[self.num_query:])
        g_camids = np.asarray(camids[self.num_query:])

        # TODO: 添加rerank的测评结果
        # m, n = qf.shape[0], gf.shape[0]
        distmat = -torch.mm(qf, gf.t()).cpu().numpy()

        # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # distmat.addmm_(1, -2, qf, gf.t())
        # distmat = distmat.numpy()
        cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
        self.logger.info(f"Test Results - Epoch: {self.current_epoch}")
        self.logger.info(f"mAP: {mAP:.1%}")
        for r in [1, 5, 10]:
            self.logger.info(f"CMC curve, Rank-{r:<3}:{cmc[r - 1]:.1%}")
        
        self.writer.add_scalar('rank1', cmc[0], self.global_step)
        self.writer.add_scalar('mAP', mAP, self.global_step)
        metric_dict = {'rank1': cmc[0], 'mAP': mAP}
        # convert to train mode
        self.model.train()
        return metric_dict

    def train(self):
        self.on_train_begin()
        for epoch in range(self.start_epoch,self.max_epochs):
            self.on_epoch_begin()
            batch = self.tng_prefetcher.next()
            while batch[0] is not None:
                self.training_step(batch)
                batch = self.tng_prefetcher.next()
            self.on_epoch_end()
            if (epoch+1) % self.eval_period == 0:
                metric_dict = self.test()
                if metric_dict['mAP'] > self.best_mAP:
                    is_best = True
                    self.best_mAP = metric_dict['mAP']
                else:
                    is_best = False

                # always save the last checkpoint as the best
                is_best = True
                self.save_checkpoint(is_best)

            torch.cuda.empty_cache()

    def save_checkpoint(self, is_best):
        if self.use_dp:
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()
        
        # TODO: add optimizer state dict and lr scheduler
        filepath = os.path.join(self.model_save_dir, f'model_epoch{self.current_epoch}.pth')
        torch.save(state_dict, filepath)
        # saving
        optpath = os.path.join(self.model_save_dir, f'optimizer_epoch{self.current_epoch}.pth')
        opt_dict = {}
        opt_dict['optimizer'] = self.opt
        opt_dict['lr_scheduler'] = self.lr_sched
        opt_dict['epoch'] = self.current_epoch
        torch.save(opt_dict, optpath)

        if is_best:
            best_filepath = os.path.join(self.model_save_dir, 'model_best.pth')
            shutil.copyfile(filepath, best_filepath)

    def load_checkpoint(self, checkpoint_path, with_optimizer=True):
        ## load weights
        self.logger.info('Loading checkpoints from ' + checkpoint_path)
        state_dict = torch.load(checkpoint_path)

        # remove missing keys
        new_state_dict = state_dict.copy()
        for k in state_dict:
            if not k in self.model.state_dict():
                new_state_dict.pop(k)
                self.logger.info(f'Remove key {k} from checkpoint.')
        state_dict = new_state_dict

        if self.use_dp:
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)

        ## load optimizer
        if with_optimizer:
            opt_path = checkpoint_path.replace('model_epoch', 'optimizer_epoch')
            self.logger.info('Loading optimizer from ' + opt_path)
            opt_dict = torch.load(opt_path)
            self.opt = opt_dict['optimizer']
            self.lr_sched = opt_dict['lr_scheduler']
            self.start_epoch = opt_dict['epoch']
            self.current_epoch = opt_dict['epoch']
            #print('start_epoch', self.start_epoch)
