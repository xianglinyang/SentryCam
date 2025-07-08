#!/bin/sh

DATA_PATH="/data2/xianglin/SentryCam"
CODE_REPO="/home/xianglin/git_space/timm_training_dynamics"

# ---------------------------------------ResNet------------------------------------------
MODEL_NAME="ResNet"

LOG_DIR="${DATA_PATH}/${MODEL_NAME}_CIFAR10_NONE_CF"
mkdir -p ${LOG_DIR}
mkdir -p ${LOG_DIR}/config
python -m case_study.catastrophic_forgetting.main --log_dir $LOG_DIR --cl_strategy none --device cuda:1 1> ${LOG_DIR}/log 2>&1
python config.py -p ${LOG_DIR}/config/tdvi.yaml
cp case_study/catastrophic_forgetting/models/resnet.py ${LOG_DIR}/Model/model.py

# ---------------------------------------ER------------------------------------------
MODEL_NAME="ResNet"

LOG_DIR="${DATA_PATH}/${MODEL_NAME}_CIFAR10_ER_CF"
mkdir -p ${LOG_DIR}
mkdir -p ${LOG_DIR}/config
python -m case_study.catastrophic_forgetting.main --log_dir $LOG_DIR --cl_strategy er --device cuda:1 1> ${LOG_DIR}/log 2>&1
python config.py -p ${LOG_DIR}/config/tdvi.yaml
cp case_study/catastrophic_forgetting/models/resnet.py ${LOG_DIR}/Model/model.py

# ---------------------------------------EWC------------------------------------------
MODEL_NAME="ResNet"

LOG_DIR="${DATA_PATH}/${MODEL_NAME}_CIFAR10_EWC_CF"
mkdir -p ${LOG_DIR}
mkdir -p ${LOG_DIR}/config
python -m case_study.catastrophic_forgetting.main --log_dir $LOG_DIR --cl_strategy ewc --device cuda:1
python config.py -p ${LOG_DIR}/config/tdvi.yaml
cp case_study/catastrophic_forgetting/models/resnet.py ${LOG_DIR}/Model/model.py


