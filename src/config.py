import os

# Training Hyperparameters
NUM_CLASSES         = 200
BATCH_SIZE          = 2048
VAL_EVERY_N_EPOCH   = 1

NUM_EPOCHS          = 40
# OPTIMIZER_PARAMS    = {'type': 'AdamW', 'lr': 0.005, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay' : 0}
OPTIMIZER_PARAMS    = {'type': 'SGD', 'lr': 0.005, 'momentum': 0.9}
# SCHEDULER_PARAMS    = {'type': 'MultiStepLR', 'milestones': [60, 90], 'gamma': 0.2}
SCHEDULER_PARAMS    = {
    'type': 'CosineAnnealingWarmRestarts',
    'T_0': 10,  # Number of iterations for the first restart
    'T_mult': 2,  # A factor increases T_i after a restart
    'eta_min': 0, # Minimum learning rate
}

# SCHEDULER_PARAMS = {
#     'type': 'CyclicLR',
#     'base_lr': 0.001,
#     'max_lr': 0.01,
#     'step_size_up': 20,
#     'mode': 'triangular'
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
MODEL_NAME          = 'resnet18'

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
