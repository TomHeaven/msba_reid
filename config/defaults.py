from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = 'baseline'
_C.MODEL.DIST_BACKEND = 'dp'
# Model backbone
_C.MODEL.BACKBONE = 'resnet50'
# Last stride for backbone
_C.MODEL.LAST_STRIDE = 1
# If use IBN block
_C.MODEL.WITH_IBN = False
# Global Context Block configuration
_C.MODEL.STAGE_WITH_GCB = (False, False, False, False)
_C.MODEL.GCB = CN()
_C.MODEL.GCB.ratio = 1./16.
# If use imagenet pretrain model
_C.MODEL.PRETRAIN = True
# Pretrain model path
_C.MODEL.PRETRAIN_PATH = ''
# Checkpoint for continuing training
_C.MODEL.CHECKPOINT = ''
_C.MODEL.VERSION = ''
_C.MODEL.OF_START_EPOCH = 33 # only used for OF module
_C.MODEL.USE_PARTS = 2

#
# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [256, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [256, 128]
# Random probability for image horizontal flip
_C.INPUT.DO_FLIP = True
_C.INPUT.FLIP_PROB = 0.5
############## Modified by TAN ############################
# 这里的值影响 data/prefetcher.py和 data/transforms/random_erase.py

# 因为原实现将BGR数据当成了RGB数据，纠正后的代码这里参数要传 RGB
#_C.INPUT.PIXEL_MEAN = [0.406, 0.456, 0.485] # 原来训练的模型使用此参数(R<G<B)
#_C.INPUT.PIXEL_STD = [0.225, 0.224, 0.229]  # 原来训练的模型使用此参数(RGB)

# 新模型直接使用BGR参数。 （初赛）
# Values to be used for image normalization
#_C.INPUT.PIXEL_MEAN = [0.213, 0.183, 0.097] # in order B>G>R
# Values to be used for image normalization
#_C.INPUT.PIXEL_STD = [0.221, 0.165, 0.176]  # in order BGR


# 新模型直接使用BGR参数。（复赛）
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.2493, 0.2129, 0.167] # in order B>G>R
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.2679, 0.2071, 0.2105]  # in order BGR

##########################################################
# Value of padding size
_C.INPUT.DO_PAD = True
_C.INPUT.PADDING_MODE = 'constant'
_C.INPUT.PADDING = 10
# Random lightning and contrast change 
_C.INPUT.DO_LIGHTING = False
_C.INPUT.MAX_LIGHTING = 0.2
_C.INPUT.P_LIGHTING = 0.75
# Random erasing
_C.INPUT.DO_RE = True
_C.INPUT.RE_PROB = 0.5

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training
_C.DATASETS.NAMES = ("market1501",)
# List of the dataset names for testing
_C.DATASETS.TEST_NAMES = "market1501"
_C.DATASETS.ROOT_DIR = ('../data')
_C.DATASETS.PRELOAD_IMAGE = False    # 是否将图像加载到内存

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'softmax'
# Number of instance for each person
_C.DATALOADER.NUM_INSTANCE = 4
_C.DATALOADER.NUM_WORKERS = 4 # old 8
_C.DATALOADER.MAX_INSTANCE = 50

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.DIST = False

_C.SOLVER.OPT = "adam"
_C.SOLVER.TRIPLET_ONLY = False
_C.SOLVER.RHO = 1.6e-3
_C.SOLVER.LOSSTYPE = ("softmax",)

_C.SOLVER.MAX_EPOCHS = 50

_C.SOLVER.BASE_LR = 3e-4
_C.SOLVER.BIAS_LR_FACTOR = 1

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.MARGIN = 0.3   # old 0.3；0.35, 0.4性能下降

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (30, 55)

_C.SOLVER.WARMUP_FACTOR = 0.1
_C.SOLVER.WARMUP_ITERS = 10
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.LOG_INTERVAL = 30
_C.SOLVER.EVAL_PERIOD = 50
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 64
_C.SOLVER.LABEL_SMOOTH = False # Previous False

_C.MODEL.FINE_TUNE = False # if True, resnet model only return bn_feat in training

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 128
_C.TEST.NORM = True
_C.TEST.WEIGHT = ""
_C.TEST.DISTMAT1 = ""
_C.TEST.DISTMAT2 = ""
_C.TEST.DISTMAT3 = ""
_C.TEST.DISTMAT4 = ""
_C.TEST.DISTMAT5 = ""
_C.TEST.DISTMAT6 = ""
_C.TEST.DISTMAT7 = ""
_C.TEST.DISTMAT8 = ""
_C.TEST.DISTMAT9 = ""
_C.TEST.DISTMAT10 = ""
_C.TEST.DISTMAT11 = ""
_C.TEST.DISTMAT12 = ""
_C.TEST.DISTMAT13 = ""
_C.TEST.DISTMAT14 = ""
_C.TEST.DISTMAT15 = ""
_C.TEST.DISTMAT16 = ""
_C.TEST.DISTMAT17 = ""
_C.TEST.DISTMAT18 = ""

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "logs/"
