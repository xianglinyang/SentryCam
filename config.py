from yacs.config import CfgNode as CN
'''
Parameters need to be rewrite:
- CLASSES
- DATASET
- EPOCH_START
- EPOCH_PERIOD
- EPOCH_END
- TRAINING.NET
- TRAINING.train_num
- TRAINING.test_num
'''
default_C = CN()

default_C.SETTING = "normal"
default_C.DATASET="CIFAR10"
default_C.CLASSES = list(range(10))
default_C.GPU = 1
default_C.EPOCH_START=1
default_C.EPOCH_PERIOD=1
default_C.EPOCH_END=35
default_C.EPOCH_NAME="Epoch"

# Training
default_C.TRAINING = CN()
default_C.TRAINING.NET="resnet18"
default_C.TRAINING.train_num=50000
default_C.TRAINING.test_num=10000

# Visualization
## basic
default_C.VISUALIZATION = CN()
default_C.VISUALIZATION.METRIC="euclidean"
default_C.VISUALIZATION.PREPROCESS=1
default_C.VISUALIZATION.SAVE_BATCH_SIZE=5000
default_C.VISUALIZATION.LAMBDA=1.0
default_C.VISUALIZATION.LAMBDA1=1.0
default_C.VISUALIZATION.LAMBDA2=0.3
default_C.VISUALIZATION.ENCODER_DIMS=[512,256,128,64,32,2]
default_C.VISUALIZATION.DECODER_DIMS=[2,32,64,128,256,512]
# default_C.VISUALIZATION.ENCODER_DIMS=[400,256,128,64,32,2]
# default_C.VISUALIZATION.DECODER_DIMS=[2,32,64,128,256,400]
default_C.VISUALIZATION.VIS_MODEL="cnAE"
default_C.VISUALIZATION.N_NEIGHBORS=15
default_C.VISUALIZATION.MAX_EPOCH=20
default_C.VISUALIZATION.S_N_EPOCHS=3
default_C.VISUALIZATION.T_N_EPOCHS=2
default_C.VISUALIZATION.PATIENT=3
default_C.VISUALIZATION.RESOLUTION=300
default_C.VISUALIZATION.INIT_NUM=300
default_C.VISUALIZATION.ALPHA=0.0
default_C.VISUALIZATION.BETA=0.1
default_C.VISUALIZATION.Delay=1
default_C.VISUALIZATION.Delay_time=32
## Boundary
default_C.VISUALIZATION.BOUNDARY = CN()
default_C.VISUALIZATION.BOUNDARY.B_N_EPOCHS=0
default_C.VISUALIZATION.BOUNDARY.L_BOUND=0.4
 

def get_cfg_defaults():
  return default_C.clone()


def update_cfg(l):
  cfg = get_cfg_defaults()
  cfg.merge_from_list(l)
  return cfg


def save_cfg(cfg, path):
  with open(path, "w") as f:
      f.write(cfg.dump())


def load_cfg(filepath):
   cfg = get_cfg_defaults()
   cfg.merge_from_file(filepath)
   return cfg

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--path", "-p", type=str, default="config.yaml")
  args = parser.parse_args()
  path = args.path

  cfg = get_cfg_defaults()
  save_cfg(cfg, path)

