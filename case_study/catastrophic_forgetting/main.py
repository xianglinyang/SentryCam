import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random
import argparse

from case_study.catastrophic_forgetting.models.resnet import resnet18
from writer.summary_writer import SummaryWriter


def get_cifar10_subset_loader(dataset, classes, batch_size=128):
    idxs = [i for i, target in enumerate(dataset.targets) if target in classes]
    return DataLoader(Subset(dataset, idxs), batch_size=batch_size, shuffle=True), idxs

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    acc = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            acc += (output.argmax(dim=1) == target).sum().item()
    return total_loss / len(loader.dataset), acc / len(loader.dataset)

class EWC:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.fisher_matrix = {}
        self.optimal_weights = {}

    def compute_fisher_information(self, dataloader: DataLoader):
        print("Computing Fisher Information Matrix for EWC...")
        self.model.eval()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.fisher_matrix[name] = torch.zeros_like(param.data)
        
        for data, target in dataloader:
            data, target = data.to(self.device), target.to(self.device)
            self.model.zero_grad()
            output = self.model(data)
            log_likelihood = torch.nn.functional.log_softmax(output, dim=1)
            sampled_labels = torch.multinomial(torch.exp(log_likelihood), 1).squeeze()
            loss = torch.nn.functional.nll_loss(log_likelihood, sampled_labels)
            loss.backward()
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.fisher_matrix[name] += param.grad.data.clone().pow(2)
        
        for name in self.fisher_matrix:
            self.fisher_matrix[name] /= len(dataloader.dataset)
        print("Fisher Information Matrix computed.")

    def save_optimal_weights(self):
        print("Saving optimal weights for EWC...")
        self.optimal_weights = {name: param.data.clone() for name, param in self.model.named_parameters() if param.requires_grad}

    def penalty(self) -> torch.Tensor:
        if not self.fisher_matrix: return torch.tensor(0.0, device=self.device)
        ewc_loss = 0.0
        for name, param in self.model.named_parameters():
            if name in self.fisher_matrix:
                ewc_loss += (self.fisher_matrix[name] * (param - self.optimal_weights[name]).pow(2)).sum()
        return ewc_loss

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size, self.buffer, self.idx = buffer_size, [], 0
    def add(self, data, targets):
        for i in range(data.size(0)):
            item = (data[i], targets[i])
            if len(self.buffer) < self.buffer_size: self.buffer.append(item)
            else: self.buffer[self.idx] = item; self.idx = (self.idx + 1) % self.buffer_size
    def sample(self, batch_size):
        if not self.buffer: return None, None
        samples = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        data, targets = zip(*samples)
        return torch.stack(data), torch.tensor(targets)

def get_replayed_batch(new_task_batch, replay_buffer, replay_batch_size):
    data_new, targets_new = new_task_batch
    if replay_buffer and replay_batch_size > 0:
        data_replay, targets_replay = replay_buffer.sample(replay_batch_size)
        if data_replay is not None:
            return torch.cat([data_new, data_replay.to(data_new.device)]), \
                   torch.cat([targets_new, targets_replay.to(targets_new.device)])
    return data_new, targets_new


# --- MAIN SCENARIO RUNNER ---
def run_split_cifar10_scenario(device, cl_strategy='none', ewc_lambda=5000.0, replay_buffer_size=500, replay_batch_proportion=0.25, writer=None):
    strategy_name = {'none': 'Baseline', 'ewc': 'EWC', 'er': 'ER'}[cl_strategy]
    print(f"\n{'='*20}\nRunning Split CIFAR-10 with Strategy: {strategy_name}\n{'='*20}")

    # Setup
    CIFAR10_TRAIN_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_TRAIN_STD = (0.2471, 0.2435, 0.2616)
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD),
        ]
    )
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD)])

    train_ds = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_ds = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    record_train_ds = datasets.CIFAR10(root='./data', train=True, download=True, transform=test_transform)
    
    t1_classes, t2_classes = [0, 1, 2, 3, 4], [5, 6, 7, 8, 9]
    train_loader_t1, t1_idxs = get_cifar10_subset_loader(train_ds, t1_classes)
    test_loader_t1, t1_test_idxs = get_cifar10_subset_loader(test_ds, t1_classes)
    train_loader_t2, t2_idxs = get_cifar10_subset_loader(train_ds, t2_classes)
    test_loader_t2, t2_test_idxs = get_cifar10_subset_loader(test_ds, t2_classes)
    record_trainloader = DataLoader(record_train_ds, batch_size=128, shuffle=False)
    testloader = DataLoader(test_ds, batch_size=100, shuffle=False)

    # >>>>>>>>>>Record Data
    writer.add_training_data(record_trainloader) # use test_transform
    writer.add_testing_data(testloader)
    # <<<<<<<<<<Record Data
    
    model = resnet18().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    
    # Initialize CL strategy if any
    ewc, replay_buffer = None, None
    if cl_strategy == 'er':
        replay_buffer = ReplayBuffer(replay_buffer_size)

    # --- Phase 1: Train on Task 1 ---
    num_t1_epochs = 15
    print(f"--- Phase 1: Training on Task 1 for {num_t1_epochs} epochs ---")
    for epoch in range(1, num_t1_epochs + 1):
        model.train()
        for data, target in train_loader_t1:
            if replay_buffer: replay_buffer.add(data.cpu(), target.cpu())
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad(); model(data); loss = criterion(model(data), target); loss.backward(); optimizer.step()
        
        model.eval()
        t1_loss, t1_acc = evaluate(model, test_loader_t1, device)
        t2_loss, t2_acc = evaluate(model, test_loader_t2, device)
        print(f"Epoch {epoch}: Task 1 Val Loss: {t1_loss:.4f}, Task 1 Val Acc: {t1_acc:.4f}")
        print(f"Epoch {epoch}: Task 2 Val Loss: {t2_loss:.4f}, Task 2 Val Acc: {t2_acc:.4f}")

        # >>>>>>>>>>record checkpoint for every 10 epochs
        writer.add_checkpoint_data(model.state_dict(), t1_idxs, epoch, epoch-1)
        # <<<<<<<<<<record checkpoint for every 10 epochs

    if cl_strategy == 'ewc':
        ewc = EWC(model, device)
        ewc.compute_fisher_information(train_loader_t1)
        ewc.save_optimal_weights()

    # --- Phase 2: Train on Task 2 ---
    num_t2_epochs = 20
    replay_batch_size = int(train_loader_t2.batch_size * replay_batch_proportion)
    print(f"--- Phase 2: Training on Task 2 for {num_t2_epochs} epochs ---")
    for epoch in range(num_t1_epochs + 1, num_t1_epochs + num_t2_epochs + 1):
        model.train()
        for new_batch in train_loader_t2:
            data, target = get_replayed_batch(new_batch, replay_buffer, replay_batch_size) if cl_strategy == 'er' else new_batch
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            if ewc: loss += ewc_lambda * ewc.penalty()
            loss.backward()
            optimizer.step()

        # >>>>>>>>>>record checkpoint for every 10 epochs
        writer.add_checkpoint_data(model.state_dict(), t2_idxs, epoch, epoch-1)
        # <<<<<<<<<<record checkpoint for every 10 epochs
            
        model.eval()
        t1_loss, t1_acc = evaluate(model, test_loader_t1, device)   
        t2_loss, t2_acc = evaluate(model, test_loader_t2, device)
        print(f"Epoch {epoch}: Task 1 Val Loss: {t1_loss:.4f}, Task 1 Val Acc: {t1_acc:.4f}")
        print(f"Epoch {epoch}: Task 2 Val Loss: {t2_loss:.4f}, Task 2 Val Acc: {t2_acc:.4f}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--cl_strategy', type=str, default='none', choices=['none', 'ewc', 'er'])
    parser.add_argument('--device', type=str, default='cuda:1')

    args = parser.parse_args()

    DEVICE = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log_dir = args.log_dir
    writer = SummaryWriter(log_dir)
    
    # Run all three scenarios to collect their histories
    run_split_cifar10_scenario(DEVICE, cl_strategy=args.cl_strategy, writer=writer)