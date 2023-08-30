########################################################################################################################
#                                                          IMPORT                                                      #
########################################################################################################################
import torch
import sys
import os
import json
import time
import numpy as np
import argparse

from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from umap.umap_ import find_ab_params

from singleVis.custom_weighted_random_sampler import CustomWeightedRandomSampler
from singleVis.vis_models import vis_models as vmodels
from singleVis.losses import UmapLoss, ReconstructionLoss, SingleVisLoss, TemporalEdgeLoss, splittDVILoss
from singleVis.edge_dataset import DataHandler, DVIDataHandler, SplitTemporalDataHandler
from singleVis.trainer import SingleVisTrainer, SplitTemporalTrainer
from singleVis.data import NormalDataProvider
from singleVis.spatial_edge_constructor import SplitSpatialTemporalEdgeConstructor, SingleEpochSpatialEdgeConstructor
from singleVis.projector import DVIProjector
from singleVis.eval.evaluator import Evaluator
from singleVis.visualizer import visualizer

########################################################################################################################
#                                                     DVI PARAMETERS                                                   #
########################################################################################################################
"""DVI with semantic temporal edges"""
VIS_METHOD = "tDVI"

########################################################################################################################
#                                                     LOAD PARAMETERS                                                  #
########################################################################################################################
parser = argparse.ArgumentParser(description='Process hyperparameters...')
parser.add_argument('--content_path', type=str)
args = parser.parse_args()

CONTENT_PATH = args.content_path
sys.path.append(CONTENT_PATH)
with open(os.path.join(CONTENT_PATH, "config.json"), "r") as f:
    config = json.load(f)
config = config[VIS_METHOD]

SETTING = config["SETTING"]
CLASSES = config["CLASSES"]
DATASET = config["DATASET"]
PREPROCESS = config["VISUALIZATION"]["PREPROCESS"]
GPU_ID = config["GPU"]
EPOCH_START = config["EPOCH_START"]
EPOCH_END = config["EPOCH_END"]
EPOCH_PERIOD = config["EPOCH_PERIOD"]
EPOCH_NAME = config["EPOCH_NAME"]

# Training parameter (subject model)
TRAINING_PARAMETER = config["TRAINING"]
NET = TRAINING_PARAMETER["NET"]
LEN = TRAINING_PARAMETER["train_num"]

# Training parameter (visualization model)
VISUALIZATION_PARAMETER = config["VISUALIZATION"]
# VIS_MODEL = VISUALIZATION_PARAMETER["VIS_MODEL"]
VIS_MODEL = 'baseAE'
LAMBDA1 = VISUALIZATION_PARAMETER["LAMBDA1"]
B_N_EPOCHS = VISUALIZATION_PARAMETER["BOUNDARY"]["B_N_EPOCHS"]
L_BOUND = VISUALIZATION_PARAMETER["BOUNDARY"]["L_BOUND"]
ENCODER_DIMS = VISUALIZATION_PARAMETER["ENCODER_DIMS"]
DECODER_DIMS = VISUALIZATION_PARAMETER["DECODER_DIMS"]
S_N_EPOCHS = VISUALIZATION_PARAMETER["S_N_EPOCHS"]
T_N_EPOCHS = VISUALIZATION_PARAMETER["T_N_EPOCHS"]
N_NEIGHBORS = VISUALIZATION_PARAMETER["N_NEIGHBORS"]
PATIENT = VISUALIZATION_PARAMETER["PATIENT"]
MAX_EPOCH = VISUALIZATION_PARAMETER["MAX_EPOCH"]

# VIS_MODEL_NAME = VISUALIZATION_PARAMETER["VIS_MODEL_NAME"]
# EVALUATION_NAME = VISUALIZATION_PARAMETER["EVALUATION_NAME"]
VIS_MODEL_NAME = f"{VIS_METHOD}_{VIS_MODEL}"
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
    data_provider._meta_data()
    if B_N_EPOCHS >0:
        data_provider._estimate_boundary(LEN//10, l_bound=L_BOUND)

# Define visualization models
model = vmodels[VIS_MODEL](ENCODER_DIMS, DECODER_DIMS)

# Calculate the size of trainable weights
total_bytes = 0
for name, param in model.named_parameters():
    if param.requires_grad:
        num_bytes = param.element_size() * param.nelement()
        total_bytes += num_bytes
print(f"Total Trainable Parameters Size: {total_bytes / (1024 * 1024):.2f} MB")

# Define Losses
negative_sample_rate = 5
min_dist = .1
_a, _b = find_ab_params(1.0, min_dist)

umap_loss_fn = UmapLoss(negative_sample_rate, _a, _b, repulsion_strength=1.0)
recon_loss_fn = ReconstructionLoss(beta=1.0)
temporal_loss_fn = TemporalEdgeLoss(negative_sample_rate, _a, _b, repulsion_strength=1.0)

single_loss_fn = SingleVisLoss(umap_loss_fn, recon_loss_fn, lambd=LAMBDA1)
# Define Projector
projector = DVIProjector(vis_model=model, content_path=CONTENT_PATH, vis_model_name=VIS_MODEL_NAME, epoch_name=EPOCH_NAME, device=DEVICE)

vis = visualizer(data_provider, projector, 200, "tab10")
save_dir = os.path.join(data_provider.content_path, "img")
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
evaluator = Evaluator(data_provider, projector, metric="euclidean")

# start epoch
# Define Criterion
criterion = SingleVisLoss(umap_loss_fn, recon_loss_fn, lambd=LAMBDA1)
# Define training parameters
optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=.1)
# Define Edge dataset
t0 = time.time()

spatial_cons = SingleEpochSpatialEdgeConstructor(data_provider, EPOCH_START, S_N_EPOCHS, B_N_EPOCHS, N_NEIGHBORS, metric="euclidean")
edge_to, edge_from, probs, feature_vectors, attention = spatial_cons.construct()

# Construct two dataset and train on them separately
dataset = DVIDataHandler(edge_to, edge_from, feature_vectors, attention)

n_samples = int(np.sum(S_N_EPOCHS * probs) // 1)
# chose sampler based on the number of dataset
if len(edge_to) > pow(2,24):
    sampler = CustomWeightedRandomSampler(probs, n_samples, replacement=True)
else:
    sampler = WeightedRandomSampler(probs, n_samples, replacement=True)
edge_loader = DataLoader(dataset, batch_size=1000, sampler=sampler, num_workers=4, prefetch_factor=10)

# train
trainer = SingleVisTrainer(model, criterion, optimizer, lr_scheduler,edge_loader=edge_loader, DEVICE=DEVICE)
train_epoch, time_spent = trainer.train(PATIENT, MAX_EPOCH)

save_dir = os.path.join(data_provider.model_path, "Epoch_{}".format(EPOCH_START))
trainer.save(save_dir=save_dir, file_name="{}".format(VIS_MODEL_NAME))
# save time result
##########################
# TODO tidy up
save_file = os.path.join(data_provider.model_path,"time_{}.json".format(VIS_MODEL_NAME))
if not os.path.exists(save_file):
    evaluation = dict()
else:
    f = open(save_file, "r")
    evaluation = json.load(f)
    f.close()
operation = "training"
if operation not in evaluation.keys():
    evaluation[operation] = dict()
evaluation[operation][str(EPOCH_START)] = (train_epoch, time_spent)
with open(save_file, 'w') as f:
    json.dump(evaluation, f)
##########################
def create_dataloader(dataset, n_epochs, weights, num_edges, batch_size=1000):
    n_samples = int(np.sum(n_epochs * probs) // 1)
    # chose sampler based on the number of dataset
    if num_edges > pow(2,24):
        sampler = CustomWeightedRandomSampler(weights, n_samples, replacement=True)
    else:
        sampler = WeightedRandomSampler(weights, n_samples, replacement=True)
    edge_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4, prefetch_factor=10, pin_memory=True)
    return edge_loader
#####################

save_dir = os.path.join(data_provider.content_path, "img")
vis.savefig(EPOCH_START, path=os.path.join(save_dir, "{}_{}_{}.png".format(VIS_METHOD, VIS_MODEL, EPOCH_START)))
evaluator.save_epoch_eval(EPOCH_START, 15, temporal_k=5, file_name="{}".format(EVALUATION_NAME))

# get prev_embedding
prev_embedding = projector.batch_project(EPOCH_START, data_provider.train_representation(EPOCH_START))
for iteration in range(EPOCH_START+EPOCH_PERIOD, EPOCH_END+EPOCH_PERIOD, EPOCH_PERIOD):
    # Define Criterion
    criterion = splittDVILoss(umap_loss_fn, recon_loss_fn, temporal_loss_fn)
    # Define training parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=.1)
    # Define Edge dataset

    spatial_cons = SplitSpatialTemporalEdgeConstructor(data_provider, projector, S_N_EPOCHS, B_N_EPOCHS, T_N_EPOCHS, N_NEIGHBORS, metric="euclidean")
    t0 = time.time()
    spatial_component, temporal_component = spatial_cons.construct(iteration)
    edge_to, edge_from, weights, feature_vectors, attention = spatial_component
    edge_t_to, edge_t_from, weight_t, next_data, prev_data, prev_embedded = temporal_component
    t1 = time.time()
    
    # two dataloaders for spatial and temporal datasets
    spatial_dataset = DataHandler(edge_to, edge_from, feature_vectors, attention)
    spatial_edge_loader = create_dataloader(spatial_dataset, S_N_EPOCHS, weights, len(edge_to))
    
    temporal_dataset = SplitTemporalDataHandler(edge_t_to, edge_t_from, next_data, prev_data, prev_embedded)
    temporal_edge_loader = create_dataloader(temporal_dataset, T_N_EPOCHS, weight_t, len(edge_t_to))
    print(len(spatial_edge_loader), len(temporal_edge_loader))
    ########################################################################################################################
    #                                                       TRAIN                                                          #
    ########################################################################################################################

    trainer = SplitTemporalTrainer(model, criterion, optimizer, lr_scheduler, spatial_edge_loader=spatial_edge_loader, temporal_edge_loader=temporal_edge_loader, DEVICE=DEVICE)

    train_epoch, time_spent = trainer.train(PATIENT, MAX_EPOCH)

    save_dir = os.path.join(data_provider.model_path, "Epoch_{}".format(iteration))
    trainer.save(save_dir=save_dir, file_name="{}".format(VIS_MODEL_NAME))
    trainer.record_time(save_dir=data_provider.model_path, file_name="time_{}".format(VIS_MODEL_NAME), operation="training", iteration=str(iteration), t=(train_epoch, time_spent))
    trainer.record_time(save_dir=data_provider.model_path, file_name="time_{}".format(VIS_MODEL_NAME), operation="complex", iteration=str(iteration), t=(t1-t0))

    save_dir = os.path.join(data_provider.content_path, "img")
    vis.savefig(iteration, path=os.path.join(save_dir, "{}_{}_{}.png".format(VIS_METHOD, VIS_MODEL, iteration)))
    evaluator.save_epoch_eval(iteration, 15, temporal_k=5, file_name="{}".format(EVALUATION_NAME))

    # get prev_embedding
    prev_embedding = projector.batch_project(iteration, data_provider.train_representation(iteration))
