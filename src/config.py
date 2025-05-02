import os

# Training Hyperparameters
NUM_CLASSES         = 200
BATCH_SIZE          = 512
VAL_EVERY_N_EPOCH   = 1

NUM_EPOCHS          = 40
# OPTIMIZER_PARAMS    = {'type': 'AdamW', 'lr': 1e-4, 'weight_decay' : 0.001}
OPTIMIZER_PARAMS    = {'type': 'SGD', 'lr': 0.01, 'momentum': 0.9}
SCHEDULER_PARAMS    = {'type': 'MultiStepLR', 'milestones': [30, 35], 'gamma': 0.2}
# SCHEDULER_PARAMS    = {
#     'type': 'CosineAnnealingWarmRestarts',
#     'T_0': 80,  # Number of iterations for the first restart
# }

# Dataaset
DATASET_ROOT_PATH   = 'datasets/'
NUM_WORKERS         = 16

# Augmentation
IMAGE_ROTATION      = 20
IMAGE_FLIP_PROB     = 0.5
IMAGE_NUM_CROPS     = 64
IMAGE_PAD_CROPS     = 4
IMAGE_MEAN          = [0.4802, 0.4481, 0.3975]
IMAGE_STD           = [0.2302, 0.2265, 0.2262]

# Network
# MODEL_NAME          = 'efficientnet_b0'
# MODEL_NAME          = 'resnet101'
MODEL_NAME          = 'Myefficient'
# MODEL_NAME          = 'convnext_base'
# MODEL_NAME          = 'MyNetwork'

# Compute related
ACCELERATOR         = 'gpu'
DEVICES             = [3]
PRECISION_STR       = '32-true'

# Logging
WANDB_PROJECT       = 'aue8088-pa1'
WANDB_ENTITY        = os.environ.get('WANDB_ENTITY')
WANDB_SAVE_DIR      = 'wandb/'
WANDB_IMG_LOG_FREQ  = 50
WANDB_NAME          = f'{MODEL_NAME}-B{BATCH_SIZE}-{OPTIMIZER_PARAMS["type"]}'
WANDB_NAME         += f'-{SCHEDULER_PARAMS["type"]}{OPTIMIZER_PARAMS["lr"]:.1E}'
