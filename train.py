import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)
sess_config = tf.ConfigProto()

import sys
import os

COCO_DATA = 'data/fss_cell'
MASK_RCNN_MODEL_PATH = 'lib/Mask_RCNN/'

if MASK_RCNN_MODEL_PATH not in sys.path:
    sys.path.append(MASK_RCNN_MODEL_PATH)
    
from samples.coco import coco
from samples.fss_cell import fss_cell

from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize
    
from lib import utils as siamese_utils
from lib import model as siamese_model
from lib import config as siamese_config
   
import time
import datetime
import random
import numpy as np
import skimage.io
import imgaug
import pickle
import matplotlib.pyplot as plt
from collections import OrderedDict

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

test_fold = [1]

folds = {
    'all': set(range(1, 15)),
    1: set(range(1, 15)) - set(range(1, 3)),
    2: set(range(1, 15)) - set(range(3, 6)),
    3: set(range(1, 15)) - set(range(6, 9)),
    4: set(range(1, 15)) - set(range(9, 11)),
    5: set(range(1, 15)) - set(range(11, 15)),
}

train_classes = list(folds[test_fold[0]])
test_classes = list(folds['all'] - folds[test_fold[0]])

# Load COCO/train dataset
coco_train = siamese_utils.IndexedFssCellDataset()
coco_train.load_coco(COCO_DATA, subset="train")
coco_train.prepare()
coco_train.build_indices()
coco_train.ACTIVE_CLASSES = train_classes

# Load COCO/test dataset
coco_val = siamese_utils.IndexedFssCellDataset()
coco_val.load_coco(COCO_DATA, subset="train")
coco_val.prepare()
coco_val.build_indices()
coco_val.ACTIVE_CLASSES = test_classes

class SmallTrainConfig(siamese_config.Config):
    # Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2 #12 # A 16GB GPU is required for a batch_size of 12
    NUM_CLASSES = 1 + 1
    NAME = 'small_coco'
    EXPERIMENT = 'example'
    CHECKPOINT_DIR = 'checkpoints/'
    # Adapt loss weights
    LOSS_WEIGHTS = {'rpn_class_loss': 2.0, 
                    'rpn_bbox_loss': 0.1, 
                    'mrcnn_class_loss': 2.0, 
                    'mrcnn_bbox_loss': 0.5, 
                    'mrcnn_mask_loss': 1.0}
    
class LargeTrainConfig(siamese_config.Config):
    # Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 4
    IMAGES_PER_GPU = 3 # 4 16GB GPUs are required for a batch_size of 12
    NUM_CLASSES = 1 + 1
    NAME = 'large_coco'
    EXPERIMENT = 'example'
    CHECKPOINT_DIR = 'checkpoints/'
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

# The small model trains on a single GPU and runs much faster.
# The large model is the same we used in our experiments but needs multiple GPUs and more time for training.
model_size = 'small' # or 'large'

if model_size == 'small':
    config = SmallTrainConfig()
elif model_size == 'large':
    config = LargeTrainConfig()
    
config.display()

# Create model object in inference mode.
model = siamese_model.SiameseMaskRCNN(mode="training", model_dir=MODEL_DIR, config=config)

train_schedule = OrderedDict()
train_schedule[1] = {"learning_rate": config.LEARNING_RATE, "layers": "heads"}
train_schedule[120] = {"learning_rate": config.LEARNING_RATE, "layers": "all"}
train_schedule[160] = {"learning_rate": config.LEARNING_RATE/10, "layers": "all"}

# Load weights trained on Imagenet
try: 
    model.load_latest_checkpoint(training_schedule=train_schedule)
except:
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