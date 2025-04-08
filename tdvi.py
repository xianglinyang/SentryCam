########################################################################################################################
#                                                          IMPORT                                                      #
########################################################################################################################
import torch
import sys
import os
import json
import time
import argparse
from umap.umap_ import find_ab_params

from singleVis.vis_models import vis_models as vmodels
from singleVis.losses import UmapLoss, ReconstructionLoss, SingleVisLoss, TemporalEdgeLoss, splittDVILoss
from singleVis.edge_dataset import DataHandler, DVIDataHandler, SplitTemporalDataHandler, create_dataloader
from singleVis.trainer import SingleVisTrainer, SplitTemporalTrainer
from singleVis.subsampling import DensityAwareSampling
from singleVis.data import NormalDataProvider
from singleVis.spatial_edge_constructor import SplitSpatialTemporalEdgeConstructor, SingleEpochSpatialEdgeConstructor
from singleVis.projector import DVIProjector
from singleVis.eval.evaluator import Evaluator
from singleVis.visualizer import visualizer

from config import load_cfg

'''
TODO:
1. weight in umap and temporal loss
2. density estimation
4. different negative sampling rate for umap and temporal loss
'''

########################################################################################################################
#                                                     DVI PARAMETERS                                                   #
########################################################################################################################
# """DVI with semantic temporal edges"""
VIS_METHOD = "tdvi"

########################################################################################################################
#                                                     LOAD PARAMETERS                                                  #
########################################################################################################################
parser = argparse.ArgumentParser(description='Process hyperparameters...')
parser.add_argument('--content_path', '-c', type=str)
parser.add_argument("--iteration", '-i', type=int)
parser.add_argument("--resume", '-r', type=int, default=-1)
parser.add_argument("--config_name", type=str, default="tdvi")
args = parser.parse_args()

########################################################################################################################
#                                                   SETTING PARAMETERS                                                 #
########################################################################################################################
CONTENT_PATH = args.content_path
ITERATION = args.iteration
RESUME = args.resume
VIS_METHOD = args.config_name

sys.path.append(CONTENT_PATH)
config = load_cfg(os.path.join(CONTENT_PATH, "config", f"{VIS_METHOD}.yaml"))
print(config)

SETTING = config.SETTING
CLASSES = config.CLASSES
DATASET = config.DATASET
PREPROCESS = config.VISUALIZATION.PREPROCESS
GPU_ID = config.GPU
EPOCH_START = config.EPOCH_START
EPOCH_END = config.EPOCH_END
EPOCH_PERIOD = config.EPOCH_PERIOD
EPOCH_NAME = config.EPOCH_NAME

# Training parameter (subject model)
TRAINING_PARAMETER = config.TRAINING
NET = TRAINING_PARAMETER.NET
LEN = TRAINING_PARAMETER.train_num

# Training parameter (visualization model)
VISUALIZATION_PARAMETER = config.VISUALIZATION
SAVE_BATCH_SIZE = VISUALIZATION_PARAMETER.SAVE_BATCH_SIZE
LAMBDA = VISUALIZATION_PARAMETER.LAMBDA
B_N_EPOCHS = VISUALIZATION_PARAMETER.BOUNDARY.B_N_EPOCHS
L_BOUND = VISUALIZATION_PARAMETER.BOUNDARY.L_BOUND
ENCODER_DIMS = VISUALIZATION_PARAMETER.ENCODER_DIMS
DECODER_DIMS = VISUALIZATION_PARAMETER.DECODER_DIMS
S_N_EPOCHS = VISUALIZATION_PARAMETER.S_N_EPOCHS
T_N_EPOCHS = VISUALIZATION_PARAMETER.T_N_EPOCHS
N_NEIGHBORS = VISUALIZATION_PARAMETER.N_NEIGHBORS
PATIENT = VISUALIZATION_PARAMETER.PATIENT
MAX_EPOCH = VISUALIZATION_PARAMETER.MAX_EPOCH
VIS_MODEL = VISUALIZATION_PARAMETER.VIS_MODEL
METRIC = VISUALIZATION_PARAMETER.METRIC

VIS_MODEL_NAME = f"{VIS_METHOD}"
EVALUATION_NAME = f"evaluation_{VIS_MODEL_NAME}"

# Define hyperparameters
DEVICE = torch.device("cuda:{}".format(GPU_ID) if torch.cuda.is_available() else "cpu")

import Model.model as subject_model
net = eval("subject_model.{}()".format(NET))

########################################################################################################################
#                                                    TRAINING SETTING                                                  #
########################################################################################################################
# Define data_provider
data_provider = NormalDataProvider(CONTENT_PATH, net, EPOCH_START, EPOCH_END, EPOCH_PERIOD, device=DEVICE, classes=CLASSES, epoch_name=EPOCH_NAME, verbose=1)
if PREPROCESS:
    data_provider._meta_data_single(ITERATION, batch_size=SAVE_BATCH_SIZE)
    if B_N_EPOCHS >0:
        data_provider._estimate_boundary_single(LEN//10, l_bound=L_BOUND, n_epoch=ITERATION, batch_size=SAVE_BATCH_SIZE)

# Define visualization models
model = vmodels[VIS_MODEL](ENCODER_DIMS, DECODER_DIMS)

# Define Losses
negative_sample_rate = 5
min_dist = .1
_a, _b = find_ab_params(1.0, min_dist)

umap_loss_fn = UmapLoss(negative_sample_rate, _a, _b, repulsion_strength=1.0)
recon_loss_fn = ReconstructionLoss(beta=1.0)
temporal_loss_fn = TemporalEdgeLoss(negative_sample_rate, _a, _b, repulsion_strength=1.0)
single_loss_fn = SingleVisLoss(umap_loss_fn, recon_loss_fn, lambd=LAMBDA)

# Define Projector
projector = DVIProjector(vis_model=model, content_path=CONTENT_PATH, vis_model_name=VIS_MODEL_NAME, epoch_name=EPOCH_NAME, device=DEVICE, verbose=0)

# Define Visualizer
vis = visualizer(data_provider, projector, 200, "tab10")

# Define Evaluator
evaluator = Evaluator(data_provider, projector, metric=METRIC)

# Define training parameters
optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=.1)

if ITERATION == EPOCH_START:
# Define Edge dataset
    t0 = time.time()

    sampler = DensityAwareSampling()
    spatial_cons = SingleEpochSpatialEdgeConstructor(data_provider, EPOCH_START, S_N_EPOCHS, B_N_EPOCHS, N_NEIGHBORS, metric="euclidean", sampler=sampler)
    edge_to, edge_from, probs, feature_vectors, attention = spatial_cons.construct()

    # Construct two dataset and train on them separately
    dataset = DVIDataHandler(edge_to, edge_from, feature_vectors, attention)
    edge_loader = create_dataloader(dataset, S_N_EPOCHS, probs, len(edge_to))

    # train
    trainer = SingleVisTrainer(model, single_loss_fn, optimizer, lr_scheduler, edge_loader=edge_loader, DEVICE=DEVICE)
    train_epoch, time_spent = trainer.train(PATIENT, MAX_EPOCH)

    save_dir = os.path.join(data_provider.model_path, f"{EPOCH_NAME}_{EPOCH_START}")
    trainer.save(save_dir=save_dir, file_name="{}".format(VIS_MODEL_NAME))
    trainer.record_time(data_provider.model_path, "time_{}".format(VIS_MODEL_NAME), "training", EPOCH_START, (train_epoch, time_spent))
    trainer.log(data_provider.content_path, EPOCH_START)
    
    vis.savefig(EPOCH_START, f"{VIS_METHOD}_{EPOCH_START}.png")
    evaluator.save_epoch_eval(EPOCH_START, 15, temporal_k=5, file_name="{}".format(EVALUATION_NAME))
else:
    # load resume epoch/iteration
    if RESUME < 0:
        log_path = os.path.join(CONTENT_PATH, "log.json")
        with open(log_path, "r") as f:
            curr_log = json.load(f)
        curr_log.sort()
        RESUME = curr_log[-1]
        print(f"Resuming from {RESUME} iterations...")
    
    projector.load(RESUME)

    # Define Criterion
    criterion = splittDVILoss(umap_loss_fn, recon_loss_fn, temporal_loss_fn)
    # Define training parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=.1)
    
    # Define Edge dataset
    sampler = DensityAwareSampling()
    spatial_cons = SplitSpatialTemporalEdgeConstructor(data_provider, projector, S_N_EPOCHS, B_N_EPOCHS, T_N_EPOCHS, N_NEIGHBORS, metric=METRIC, sampler=sampler)
    t0 = time.time()
    spatial_component, temporal_component = spatial_cons.construct(ITERATION, RESUME)
    edge_to, edge_from, weights, feature_vectors, attention = spatial_component
    edge_t_to, edge_t_from, weight_t, next_data, prev_data, prev_embedded, margins = temporal_component
    t1 = time.time()
    
    # two dataloaders for spatial and temporal datasets
    spatial_dataset = DataHandler(edge_to, edge_from, feature_vectors, attention)
    spatial_edge_loader = create_dataloader(spatial_dataset, S_N_EPOCHS, weights, len(edge_to))
    
    temporal_dataset = SplitTemporalDataHandler(edge_t_to, edge_t_from, next_data, prev_data, prev_embedded, margins)
    temporal_edge_loader = create_dataloader(temporal_dataset, T_N_EPOCHS, weight_t, len(edge_t_to))
    ########################################################################################################################
    #                                                       TRAIN                                                          #
    ########################################################################################################################
    trainer = SplitTemporalTrainer(model, criterion, optimizer, lr_scheduler, spatial_edge_loader=spatial_edge_loader, temporal_edge_loader=temporal_edge_loader, DEVICE=DEVICE)

    train_epoch, time_spent = trainer.train(PATIENT, MAX_EPOCH)

    save_dir = os.path.join(data_provider.model_path, "{}_{}".format(EPOCH_NAME, ITERATION))
    trainer.save(save_dir=save_dir, file_name="{}".format(VIS_MODEL_NAME))
    trainer.record_time(save_dir=data_provider.model_path, file_name="time_{}".format(VIS_MODEL_NAME), operation="training", iteration=str(ITERATION), t=(train_epoch, time_spent))
    trainer.record_time(save_dir=data_provider.model_path, file_name="time_{}".format(VIS_MODEL_NAME), operation="complex", iteration=str(ITERATION), t=(t1-t0))
    trainer.log(data_provider.content_path, ITERATION)

    vis.savefig(ITERATION, f"{VIS_METHOD}_{ITERATION}.png")
    evaluator.save_epoch_eval(ITERATION, 15, temporal_k=5, file_name="{}".format(EVALUATION_NAME))
