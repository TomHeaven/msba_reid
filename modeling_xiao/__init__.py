# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from torch import nn

# from .losses import reidLoss

def build_model(cfg, num_classes) -> nn.Module:
	if len(cfg.TEST.WEIGHT) > 0:
		print('>>>>>>>>>>>>>Load model with from pre-trained model<<<<<<<<<<<<<<<')
		print('>>>>>>>>>>>>>Only used in finetune or inference<<<<<<<<<<<<<<<')
		# from .baseline_parts_old_ft import Baseline
		from .baseline_parts_ft import Baseline
	else:
		# from .baseline_parts_old import Baseline
		from .baseline_parts import Baseline
		print('>>>>>>>>>>>>>Load model with imagenet pre-trained<<<<<<<<<<<<<<<')

	model = Baseline(
	       cfg.MODEL.BACKBONE, 
	       num_classes, 
	       cfg.MODEL.LAST_STRIDE, 
	       cfg.MODEL.WITH_IBN, 
	       cfg.MODEL.GCB, 
	       cfg.MODEL.STAGE_WITH_GCB,
	       cfg.MODEL.USE_PARTS,
	       cfg.MODEL.PRETRAIN, 
	       cfg.MODEL.PRETRAIN_PATH)
	return model

