"""
A class to record training dynamics, including:
1. loss
2. uncertainty
3. position
4. velocity
5. acceleration
6. hard samples
7. training dynamics
8.
"""
import numpy as np
import umap
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def cross_entropy(data, y, class_num=10):
    log_p = np.array([np.log(softmax(data[i])) for i in range(len(data))])
    y_onehot = np.eye(class_num)[y]
    loss = - np.sum(y_onehot * log_p, axis=1)
    return loss


class TD:
    def __init__(self, data_provider, projector, range=None) -> None:
        self.data_provider = data_provider
        self.projector = projector
        if range is None:
            self.range = (self.data_provider.s, self.data_provider.e, self.data_provider.p)
        else:
            self.range = range

    def loss_dynamics(self, train=True):
        EPOCH_START, EPOCH_END, EPOCH_PERIOD = self.range
        # epoch, num, 1
        losses = None

        for epoch in range(EPOCH_START, EPOCH_END+EPOCH_PERIOD, EPOCH_PERIOD):
            if train:
                representation = self.data_provider.train_representation(epoch)
                labels = self.data_provider.train_labels(epoch)
            else:
                representation = self.data_provider.test_representation(epoch)
                labels = self.data_provider.test_labels(epoch)

            pred = self.data_provider.get_pred(epoch, representation)
            loss = cross_entropy(pred, labels, class_num=len(self.data_provider.classes))

            if losses is None:
                losses = np.expand_dims(loss, axis=0)
            else:
                losses = np.concatenate((losses, np.expand_dims(loss, axis=0)), axis=0)
        losses = np.transpose(losses, [1,0])
        return losses
    
    def uncertainty_dynamics(self):
        EPOCH_START, EPOCH_END, EPOCH_PERIOD = self.range
        labels = self.data_provider.train_labels(EPOCH_START)

        # epoch, num, 1
        uncertainties = None

        for epoch in range(EPOCH_START, EPOCH_END+1, EPOCH_PERIOD):
            representation = self.data_provider.train_representation(epoch)
            pred = self.data_provider.get_pred(epoch, representation)
            uncertainty = pred[np.arange(len(labels)), labels]

            if uncertainties is None:
                uncertainties = np.expand_dims(uncertainty, axis=0)
            else:
                uncertainties = np.concatenate((uncertainties, np.expand_dims(uncertainty, axis=0)), axis=0)
        uncertainties = np.transpose(uncertainties, [1,0])
        return uncertainties
    
    def pred_dynamics(self):
        EPOCH_START, EPOCH_END, EPOCH_PERIOD = self.range
        # epoch, num, 1
        preds = None

        for epoch in range(EPOCH_START, EPOCH_END+1, EPOCH_PERIOD):
            representation = self.data_provider.train_representation(epoch)
            pred = self.data_provider.get_pred(epoch, representation)

            if preds is None:
                preds = np.expand_dims(pred, axis=0)
            else:
                preds = np.concatenate((preds, np.expand_dims(pred, axis=0)), axis=0)
        preds = np.transpose(preds, [1,0,2])
        return preds
    
    def dloss_dt_dynamics(self, ):
        return
    
    def position_dynamics(self):
        EPOCH_START, EPOCH_END, EPOCH_PERIOD = self.range

        # epoch, num, dims
        embeddings = None

        for epoch in range(EPOCH_START, EPOCH_END+1, EPOCH_PERIOD):
            representation = self.data_provider.train_representation(epoch)
            embedding = self.projector.batch_project(epoch, representation)
            if embeddings is None:
                embeddings = np.expand_dims(embedding, axis=0)
            else:
                embeddings = np.concatenate((embeddings, np.expand_dims(embedding, axis=0)), axis=0)
        embeddings = np.transpose(embeddings, [1,0,2])
        return embeddings
    
    def position_high_dynamics(self):
        EPOCH_START, EPOCH_END, EPOCH_PERIOD = self.range
        # epoch, num, dims
        representations = None

        for epoch in range(EPOCH_START, EPOCH_END+1, EPOCH_PERIOD):
            representation = self.data_provider.train_representation(epoch)
            if representations is None:
                representations = np.expand_dims(representation, axis=0)
            else:
                representations = np.concatenate((representations, np.expand_dims(representation, axis=0)), axis=0)
        representations = np.transpose(representations, [1,0,2])
        return representations
    
    def velocity_dynamics(self,):
        position_dynamics = self.position_dynamics()
        return position_dynamics[:, 1:, :] - position_dynamics[:, :-1, :]
    
    def velocity_high_dynamics(self,):
        position_dynamics = self.position_high_dynamics()
        return position_dynamics[:, 1:, :] - position_dynamics[:, :-1, :]

    def acceleration_dynamics(self, ):
        velocity_dynamics = self.velocity_dynamics()
        return velocity_dynamics[:, 1:, :] - velocity_dynamics[:, :-1, :]
    
    def acceleration_high_dynamics(self, ):
        velocity_dynamics = self.velocity_high_dynamics()
        return velocity_dynamics[:, 1:, :] - velocity_dynamics[:, :-1, :]
    
    def simplify_2(self, trajectories, time_step, method):
        '''Choose from
        1. PCA
        2. UMAP
        3. interpretable method
        '''
        if method == "UMAP":
            num = len(trajectories)
            trajectories = trajectories.reshape(num, -1)
            # Non-linear
            reducer = umap.UMAP(n_components=2)
            embeddings = reducer.fit_transform(trajectories)
        elif method=="PCA":
            num = len(trajectories)
            trajectories = trajectories.reshape(num, -1)
            # Linear
            reducer = PCA(n_components=2)
            embeddings = reducer.fit_transform(trajectories)
        elif method == "length":
            trajectories = trajectories.reshape(len(trajectories), time_step, -1)
            movement = np.abs(trajectories[:, 1:, :] - trajectories[:, :-1, :])
            # scalar
            # embeddings = np.linalg.norm(movement, axis=2).sum(axis=1)
            # two dimension
            embeddings = movement.sum(axis=1).squeeze()
        elif method == "avg_v":
            trajectories = trajectories.reshape(len(trajectories), time_step, -1)
            v = np.abs(trajectories[:, 1:, :] - trajectories[:, :-1, :])
            embeddings = v.mean(axis=1).squeeze()
            # embeddings = np.linalg.norm(v, axis=2).sum(axis=1)
        elif method == "max_v":
            trajectories = trajectories.reshape(len(trajectories), time_step, -1)
            v = np.abs(trajectories[:, 1:, :] - trajectories[:, :-1, :])
            idxs = np.linalg.norm(v, axis=2).argmax(axis=1)
            embeddings = v[np.arange(len(v)), idxs]
        elif method == "net_displacement":
            trajectories = trajectories.reshape(len(trajectories), time_step, -1)
            dists = (trajectories[:, -1, :] - trajectories[:, 0, :]).squeeze()
            # embeddings = np.linalg.norm(dists, axis=1).squeeze()
            embeddings = dists
        elif method == "RoG":
            trajectories = trajectories.reshape(len(trajectories), time_step, -1)
            # Radius of Gyration
            mean_t = trajectories.mean(axis=0)
            embeddings = np.abs(trajectories-mean_t).mean(axis=1)
        else:
            raise TypeError("No method Implemented!")

        return embeddings

    def simplify_1(self, trajectories, time_step, method):
        '''Choose from
        1. PCA
        2. UMAP
        3. interpretable method
        '''
        if method == "UMAP":
            num = len(trajectories)
            trajectories = trajectories.reshape(num, -1)
            # Non-linear
            reducer = umap.UMAP(n_components=1)
            embeddings = reducer.fit_transform(trajectories)
        elif method=="PCA":
            num = len(trajectories)
            trajectories = trajectories.reshape(num, -1)
            # Linear
            reducer = PCA(n_components=1)
            embeddings = reducer.fit_transform(trajectories)
        elif method == "length":
            trajectories = trajectories.reshape(len(trajectories), time_step, -1)
            movement = np.abs(trajectories[:, 1:, :] - trajectories[:, :-1, :])
            # scalar
            embeddings = np.linalg.norm(movement, axis=2).sum(axis=1)
        elif method == "avg_v":
            trajectories = trajectories.reshape(len(trajectories), time_step, -1)
            v = np.abs(trajectories[:, 1:, :] - trajectories[:, :-1, :])
            embeddings = np.linalg.norm(v, axis=2).sum(axis=1)
        elif method == "max_v":
            trajectories = trajectories.reshape(len(trajectories), time_step, -1)
            v = np.abs(trajectories[:, 1:, :] - trajectories[:, :-1, :])
            speed = np.linalg.norm(v, axis=2)
            idxs = speed.argmax(axis=1)
            embeddings = speed[np.arange(len(v)), idxs]
        elif method == "net_displacement":
            trajectories = trajectories.reshape(len(trajectories), time_step, -1)
            dists = (trajectories[:, -1, :] - trajectories[:, 0, :]).squeeze()
            embeddings = np.linalg.norm(dists, axis=1).squeeze()
        elif method == "RoG":
            trajectories = trajectories.reshape(len(trajectories), time_step, -1)
            # Radius of Gyration
            mean_t = trajectories.mean(axis=0)
            embeddings = np.linalg.norm(np.abs(trajectories-mean_t), axis=(1,2))
        else:
            raise TypeError("No method Implemented!")

        return embeddings

    def plot_ground_truth_2(self, embeddings, noise_idxs, colors, save_path=None, cmap="tab10"):
        plt.scatter(
            embeddings[:, 0],
            embeddings[:, 1],
            s=.3,
            c=colors,
            cmap=cmap
            )
        
        plt.scatter(
            embeddings[:, 0][noise_idxs],
            embeddings[:, 1][noise_idxs],
            s=.4,
            c='black')
        
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)

    def plot_ground_truth_1(self, embeddings, noise_idxs, save_path=None):
        clean_idxs = np.setxor1d(np.arange(len(embeddings)), noise_idxs) 
        plt.hist(embeddings[clean_idxs], bins=50)
        plt.hist(embeddings[noise_idxs], bins=50)

        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)
    
    def show_ground_truth_2(self, trajectories, time_step, noise_idxs, cls, method, colors=None, save_path=None, cmap="tab10"):
        EPOCH_START = self.data_provider.s
        labels = self.data_provider.train_labels(EPOCH_START)

        cls_idxs = np.argwhere(labels == cls).squeeze()
        mask = np.isin(cls_idxs, noise_idxs)

        target_trajectories =  trajectories[cls_idxs]
        
        embeddings = self.simplify_2(target_trajectories, time_step, method)

        dists = np.linalg.norm(embeddings, axis=1)
        ranking = cls_idxs[np.argsort(dists)]

        if colors is None:
            paint_colors = [0]*len(target_trajectories)
        else:
            paint_colors = colors[cls_idxs]
        self.plot_ground_truth_2(embeddings, mask, paint_colors, save_path=save_path, cmap=cmap)
        return ranking
    
    def show_ground_truth_1(self, trajectories, time_step, noise_idxs, cls, method, save_path=None):
        EPOCH_START = self.data_provider.s
        labels = self.data_provider.train_labels(EPOCH_START)

        cls_idxs = np.argwhere(labels == cls).squeeze()
        mask = np.isin(cls_idxs, noise_idxs)

        target_trajectories =  trajectories[cls_idxs]
        
        embeddings = self.simplify_1(target_trajectories, time_step, method)
        ranking = cls_idxs[np.argsort(embeddings)]
        self.plot_ground_truth_1(embeddings, mask, save_path=save_path)
        return ranking
    

    