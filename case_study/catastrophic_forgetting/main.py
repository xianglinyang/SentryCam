import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random
import argparse
import copy

from case_study.catastrophic_forgetting.models.resnet import resnet18
from writer.summary_writer import SummaryWriter


def get_cifar10_subset_loader(dataset, classes, batch_size=128):
    idxs = [i for i, target in enumerate(dataset.targets) if target in classes]
    return DataLoader(Subset(dataset, idxs), batch_size=batch_size, shuffle=True), idxs

# def evaluate(model, loader, device):
#     model.eval()
#     total_loss = 0
#     acc = 0
#     criterion = nn.CrossEntropyLoss(reduction='sum')
#     with torch.no_grad():
#         for data, target in loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             total_loss += criterion(output, target).item()
#             acc += (output.argmax(dim=1) == target).sum().item()
#     return total_loss / len(loader.dataset), acc / len(loader.dataset)

# --- REVISED EVALUATE FUNCTION (CRITICAL FIX) ---
def evaluate(model, loader, device, active_classes=None, return_acc=True):
    """
    Evaluates the model. If active_classes are provided, performs
    continual learning style evaluation by filtering logits.
    """
    model.eval()
    total_loss, correct, total = 0, 0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Use full logits for loss calculation as it's more stable
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)

            if return_acc:
                if active_classes:
                    # Continual Learning evaluation: only consider active class logits
                    output_active = output[:, active_classes]
                    pred_local = output_active.argmax(dim=1)
                    # Map local predictions (0, 1, 2...) back to original class indices
                    pred = torch.tensor([active_classes[i] for i in pred_local], device=device)
                else:
                    # Standard evaluation
                    pred = output.argmax(dim=1)
                
                correct += pred.eq(target).sum().item()
                total += data.size(0)

    avg_loss = total_loss / len(loader.dataset)
    if return_acc:
        accuracy = 100. * correct / total if total > 0 else 0
        return avg_loss, accuracy
    return avg_loss

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


# # --- REVISED SCENARIO RUNNER ---
# def run_subcluster_collapse_scenario(device, writer):
#     print(f"\n{'='*20}\nRunning 'Sub-Cluster Collapse' Scenario (Revised)\n{'='*20}")
    
#     # --- Setup ---
#     task1_classes = [0, 8, 3, 5, 2]  # airplane, ship, cat, dog, bird
#     task2_classes = [1, 4, 6, 7, 9]  # automobile, deer, frog, horse, truck
#     cat_dog_pair = [3, 5]
    
#     CIFAR10_TRAIN_MEAN = (0.4914, 0.4822, 0.4465)
#     CIFAR10_TRAIN_STD = (0.2471, 0.2435, 0.2616)
#     train_transform = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD),
#     ])
#     test_transform = transforms.Compose([
#         transforms.ToTensor(), 
#         transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD)
#     ])

#     ds_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
#     ds_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
#     ds_record = datasets.CIFAR10(root='./data', train=True, download=True, transform=test_transform)
    
#     train_loader_t1, t1_idxs = get_cifar10_subset_loader(ds_train, task1_classes)
#     test_loader_t1, _ = get_cifar10_subset_loader(ds_test, task1_classes)
#     train_loader_t2, t2_idxs = get_cifar10_subset_loader(ds_train, task2_classes)
#     test_loader_t2, _ = get_cifar10_subset_loader(ds_test, task2_classes)
#     cat_dog_test_loader, _ = get_cifar10_subset_loader(ds_test, cat_dog_pair)

#     record_trainloader = DataLoader(ds_record, batch_size=128, shuffle=False)
#     testloader = DataLoader(ds_test, batch_size=100, shuffle=False)

#     # >>>>>>>>>> Record Data
#     # Assuming the writer object is passed in and has these methods
#     writer.add_training_data(record_trainloader)
#     writer.add_testing_data(testloader)
#     # <<<<<<<<<< Record Data

#     model = resnet18().to(device) # Assuming resnet18 is available
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     criterion = nn.CrossEntropyLoss()
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
#     # --- Training Phases ---
#     num_t1_epochs, num_t2_epochs = 15, 20
#     print(f"--- Phase 1: Training on Task 1 for {num_t1_epochs} epochs ---")
#     for epoch in range(1, num_t1_epochs + num_t2_epochs + 1):
#         is_task1_phase = epoch <= num_t1_epochs
        
#         if epoch == num_t1_epochs + 1:
#             print(f"\n--- Phase 2: Training on Task 2 for {num_t2_epochs} epochs ---")
        
#         train_loader = train_loader_t1 if is_task1_phase else train_loader_t2
        
#         model.train()
#         for data, target in train_loader:
#             data, target = data.to(device), target.to(device)
#             optimizer.zero_grad()
#             loss = criterion(model(data), target)
#             loss.backward()
#             optimizer.step()
        
#         scheduler.step()
        
#         # --- Evaluation (using the corrected evaluate function) ---
#         model.eval()
#         # When evaluating a task, pass its specific classes to the evaluate function
#         loss_t1, acc_t1 = evaluate(model, test_loader_t1, device, active_classes=task1_classes)
#         loss_t2, acc_t2 = evaluate(model, test_loader_t2, device, active_classes=task2_classes)
#         _, cat_dog_acc = evaluate(model, cat_dog_test_loader, device, active_classes=cat_dog_pair)
        
#         print(f"Epoch {epoch}: T1 Acc: {acc_t1:.2f}% | T2 Acc: {acc_t2:.2f}% | Cat/Dog Acc: {cat_dog_acc:.2f}%")

#         # >>>>>>>>>> Record checkpoint
#         # This logic remains the same
#         if is_task1_phase:
#             writer.add_checkpoint_data(model.state_dict(), t1_idxs, epoch, epoch-1)
#         else:
#             writer.add_checkpoint_data(model.state_dict(), t2_idxs, epoch, epoch-1)
#         # <<<<<<<<<< Record checkpoint

# --- REVISED UNIFIED SCENARIO RUNNER ---
def run_subtle_forgetting_scenario(device, writer, method='none', 
                                   replay_buffer_size=200, replay_proportion=0.1, 
                                   distillation_lambda=1.0):
    """
    Runs the Split CIFAR-10 scenario with tunable subtle forgetting.
    Args:
        method (str): 'none' for baseline, 'replay' for Method 2, 'distillation' for Method 3.
    """
    scenario_name_map = {'none': 'Baseline (Catastrophic Forgetting)', 'replay': 'Subtle Replay', 'distillation': 'Feature Distillation'}
    print(f"\n{'='*20}\nRunning Scenario: {scenario_name_map[method]}\n{'='*20}")
    
    # --- Setup (from your code) ---
    task1_classes = [0, 8, 3, 5, 2]
    task2_classes = [1, 4, 6, 7, 9]
    cat_dog_pair = [3, 5]
    CIFAR10_TRAIN_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_TRAIN_STD = (0.2471, 0.2435, 0.2616)
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD)])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD)])
    ds_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    ds_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    ds_record = datasets.CIFAR10(root='./data', train=True, download=True, transform=test_transform)
    train_loader_t1, t1_idxs = get_cifar10_subset_loader(ds_train, task1_classes)
    test_loader_t1, _ = get_cifar10_subset_loader(ds_test, task1_classes)
    train_loader_t2, t2_idxs = get_cifar10_subset_loader(ds_train, task2_classes)
    test_loader_t2, _ = get_cifar10_subset_loader(ds_test, task2_classes)
    cat_dog_test_loader, _ = get_cifar10_subset_loader(ds_test, cat_dog_pair)
    record_trainloader = DataLoader(ds_record, batch_size=128, shuffle=False)
    testloader = DataLoader(ds_test, batch_size=100, shuffle=False)

    writer.add_training_data(record_trainloader)
    writer.add_testing_data(testloader)

    model = resnet18().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scheduler_t1 = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    # --- CL Strategy Initialization ---
    replay_buffer, model_t1 = None, None
    if method == 'replay':
        replay_buffer = ReplayBuffer(replay_buffer_size)

    # --- Training Phases ---
    num_t1_epochs, num_t2_epochs = 10, 20
    print(f"--- Phase 1: Training on Task 1 for {num_t1_epochs} epochs ---")
    for epoch in range(1, num_t1_epochs + 1):
        model.train()
        for data, target in train_loader_t1:
            if replay_buffer: replay_buffer.add(data.cpu(), target.cpu())
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad(); loss = criterion(model(data), target); loss.backward(); optimizer.step()
        scheduler_t1.step()
        
        loss_t1, acc_t1 = evaluate(model, test_loader_t1, device, active_classes=task1_classes)
        loss_t2, acc_t2 = evaluate(model, test_loader_t2, device, active_classes=task2_classes)
        loss_cat_dog, cat_dog_acc = evaluate(model, cat_dog_test_loader, device, active_classes=cat_dog_pair)
        print(f"Epoch {epoch}: T1 Loss: {loss_t1:.4f} | T1 Acc: {acc_t1:.2f}% | T2 Loss: {loss_t2:.4f} | T2 Acc: {acc_t2:.2f}% | Cat/Dog Loss: {loss_cat_dog:.4f} | Cat/Dog Acc: {cat_dog_acc:.2f}%")
        writer.add_checkpoint_data(model.state_dict(), t1_idxs, epoch, epoch-1)
    
    # Save a copy of the model after Task 1 for the distillation method
    if method == 'distillation':
        model_t1 = copy.deepcopy(model).eval()

    print("\nRe-initializing optimizer for Task 2...")
    optimizer = optim.Adam(model.parameters(), lr=0.0001) # Start with a lower LR for warm-up
    
    # **FIX 2: Use a dedicated scheduler for Phase 2 with warm-up.**
    # This scheduler will increase LR for a few epochs, then decrease it.
    warmup_epochs = 3
    main_scheduler_t2 = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
    scheduler_t2 = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: min(1.0, (epoch + 1) / warmup_epochs)
    )

    print(f"\n--- Phase 2: Training on Task 2 for {num_t2_epochs} epochs ---")
    replay_batch_size = int(train_loader_t2.batch_size * replay_proportion)

    for epoch in range(num_t1_epochs + 1, num_t1_epochs + num_t2_epochs + 1):
        model.train()
        train_loader = train_loader_t2
        for data, target in train_loader:
            # The 'data' and 'target' from the loader are on the CPU.
            # We will handle device placement based on the method.
            
            optimizer.zero_grad()
            total_loss = 0

            # --- METHOD-SPECIFIC LOSS CALCULATION WITH CORRECT DEVICE PLACEMENT ---
            if method == 'replay':
                # `get_replayed_batch` will combine the CPU tensors.
                # `data` and `target` from the loader are passed as a tuple.
                data, target = get_replayed_batch((data, target), replay_buffer, replay_batch_size)
                # *** FIX: Move the final combined batch to the correct device ***
                data, target = data.to(device), target.to(device)
                total_loss = criterion(model(data), target)
            
            elif method == 'distillation':
                # *** FIX: Move data and target to device here ***
                data, target = data.to(device), target.to(device)
                current_features = model.feature(data)
                # output = model.fc2(model.relu(model.fc1(current_features))) # Assuming ResNet structure
                output = model(data)
                primary_loss = criterion(output, target)
                with torch.no_grad():
                    target_features = model_t1.feature(data)
                distill_loss = torch.nn.functional.mse_loss(current_features, target_features)
                total_loss = primary_loss + distillation_lambda * distill_loss
            
            else: # 'none'
                # *** FIX: Move data and target to device here ***
                data, target = data.to(device), target.to(device)
                total_loss = criterion(model(data), target)
            
            total_loss.backward()
            optimizer.step()
            
        # Update the correct scheduler based on the current epoch in Phase 2
        if epoch < warmup_epochs:
            scheduler_t2.step() # Use warmup scheduler
        else:
            # After warmup, switch to the main scheduler
            # We need to call it relative to the end of warmup
            if epoch == warmup_epochs:
                # One-time transition to the main scheduler's state
                for group in optimizer.param_groups:
                    group['lr'] = 0.001 # Reset to the main LR
            main_scheduler_t2.step()
        
        # --- Evaluation ---
        loss_t1, acc_t1 = evaluate(model, test_loader_t1, device, active_classes=task1_classes)
        loss_t2, acc_t2 = evaluate(model, test_loader_t2, device, active_classes=task2_classes)
        loss_cat_dog, cat_dog_acc = evaluate(model, cat_dog_test_loader, device, active_classes=cat_dog_pair)
        print(f"Epoch {epoch}: T1 Loss: {loss_t1:.4f} | T1 Acc: {acc_t1:.2f}% | T2 Loss: {loss_t2:.4f} | T2 Acc: {acc_t2:.2f}% | Cat/Dog Loss: {loss_cat_dog:.4f} | Cat/Dog Acc: {cat_dog_acc:.2f}%")
        writer.add_checkpoint_data(model.state_dict(), t2_idxs, epoch, epoch-1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='./logs')
    # parser.add_argument('--cl_strategy', type=str, default='none', choices=['none', 'ewc', 'er'])
    parser.add_argument('--device', type=str, default='cuda:1')

    args = parser.parse_args()

    DEVICE = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log_dir = args.log_dir
    writer = SummaryWriter(log_dir)
    
    # Run all three scenarios to collect their histories
    # run_split_cifar10_scenario(DEVICE, cl_strategy=args.cl_strategy, writer=writer)
    run_subtle_forgetting_scenario(DEVICE, method='distillation', writer=writer)