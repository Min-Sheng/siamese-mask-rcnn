import os
import tensorflow as tf
from keras import backend as K
tf.logging.set_verbosity(tf.logging.INFO)
config = tf.ConfigProto(device_count={'GPU':4}, intra_op_parallelism_threads=2, inter_op_parallelism_threads=2, allow_soft_placement=True)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

import sys
import os
import glob

COCO_DATA = '../data/fss_cell'
MASK_RCNN_MODEL_PATH = '../lib/Mask_RCNN/'
SIAMESE_MASK_RCNN_PATH = '../'

if MASK_RCNN_MODEL_PATH not in sys.path:
    sys.path.append(MASK_RCNN_MODEL_PATH)
if SIAMESE_MASK_RCNN_PATH not in sys.path:
    sys.path.append(SIAMESE_MASK_RCNN_PATH)
    
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize
    
from lib import utils as siamese_utils
from lib import model as siamese_model
from lib import config as siamese_config
from collections import OrderedDict
    
import time
import datetime
import random
import numpy as np
import skimage.io
import imgaug
import pickle

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

index = 4

class TrainConfig(siamese_config.Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 3
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 1
    NAME = 'fss_cell'
    EXPERIMENT = 'i{}'.format(index)
    CHECKPOINT_DIR = '../checkpoints/'
    # Reduced image sizes
    TARGET_MAX_DIM = 192
    TARGET_MIN_DIM = 150
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024
    # Reduce model size
    FPN_CLASSIF_FC_LAYERS_SIZE = 1024
    FPN_FEATUREMAPS = 256
    # Reduce number of rois at all stages
    RPN_ANCHOR_STRIDE = 1
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000
    TRAIN_ROIS_PER_IMAGE = 200
    DETECTION_MAX_INSTANCES = 100
    MAX_GT_INSTANCES = 100
    # Adapt NMS Threshold
    DETECTION_NMS_THRESHOLD = 0.5
    # Adapt loss weights
    LOSS_WEIGHTS = {'rpn_class_loss': 2.0, 
                    'rpn_bbox_loss': 0.1, 
                    'mrcnn_class_loss': 2.0, 
                    'mrcnn_bbox_loss': 0.5, 
                    'mrcnn_mask_loss': 1.0}
    
config = TrainConfig()
config.display()

exp_dir = os.path.join(ROOT_DIR, "{}_{}".format(config.NAME.lower(), config.EXPERIMENT.lower()))

folds = {
    'all': set(range(1, 15)),
    1: set(range(1, 15)) - set(range(1, 3)),
    2: set(range(1, 15)) - set(range(3, 6)),
    3: set(range(1, 15)) - set(range(6, 9)),
    4: set(range(1, 15)) - set(range(9, 11)),
    5: set(range(1, 15)) - set(range(11, 15)),
}

train_classes = np.array(list(folds[index]))
test_classes =  np.array(list(folds['all'] - folds[index]))

# Load COCO/train dataset
coco_train = siamese_utils.IndexedFssCellDataset()
coco_train.load_coco(COCO_DATA, subset="train_val")
coco_train.prepare()
coco_train.build_indices()
coco_train.ACTIVE_CLASSES = train_classes

# Load COCO/val dataset
coco_val = siamese_utils.IndexedFssCellDataset()
coco_val.load_coco(COCO_DATA, subset="test")
coco_val.prepare()
coco_val.build_indices()
coco_val.ACTIVE_CLASSES = test_classes

# Create model object in inference mode.
model = siamese_model.SiameseMaskRCNN(mode="training", model_dir=MODEL_DIR, config=config)

train_schedule = OrderedDict()
train_schedule[1] = {"learning_rate": config.LEARNING_RATE, "layers": "heads"}
train_schedule[10] = {"learning_rate": config.LEARNING_RATE, "layers": "all"}
train_schedule[10] = {"learning_rate": config.LEARNING_RATE/10, "layers": "all"}

# Load weights trained on Imagenet
model.load_imagenet_weights(pretraining='imagenet-687')

for epochs, parameters in train_schedule.items():
    print("")
    print("training layers {} until epoch {} with learning_rate {}".format(parameters["layers"], 
                                                                          epochs, 
                                                                          parameters["learning_rate"]))
    model.train(coco_train, coco_val, 
                learning_rate=parameters["learning_rate"], 
                epochs=epochs, 
                layers=parameters["layers"])
session.close()