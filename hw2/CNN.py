import numpy as np
import torch
import torchvision   
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torchvision.transforms.transforms import RandomAffine, RandomApply, RandomHorizontalFlip, RandomRotation, RandomVerticalFlip
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import os
from PIL import Image
import matplotlib.pyplot as plt


# use custom dataset to load classification test data
# return the tensor of the ith image in order_file with index i
class ClassificationTestDataset(Dataset):
    def __init__(self, datadir, order_file):
        img_list = []
        with open(order_file) as f:
          for testfile in f.readlines():
            testfile = testfile.rstrip()
            img = Image.open(os.path.join(datadir, testfile))
            img_list.append(torchvision.transforms.ToTensor()(img))
        self.img_list = torch.stack(img_list, dim=0)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        return self.img_list[index]


# use custom dataset to load verification data
# return the tensors of the ith image pair in order_file with index i, or along with the label if in validation
class VerificationDataset(Dataset):
    def __init__(self, datadir, order_file, mode):
        self.mode = mode
        img_list = []
        if(self.mode == "validation"):
            label_list = []
        with open(order_file) as f:
            for line in f.readlines():
                if(self.mode == "validation"):
                    img1, img2, label = line.split()
                    img_list.append([img1, img2])
                    label_list.append(int(label))
                elif(self.mode == "test"):
                    img1, img2 = line.split()
                    img_list.append([img1, img2])
        self.img_list = img_list
        if(self.mode == "validation"):
            self.label_list = torch.tensor(label_list)

    def __len__(self):
        return len(self.img_list)

    # convert file name to data here to save RAM
    def __getitem__(self, index):
        img1, img2 = self.img_list[index]
        img1 = torchvision.transforms.ToTensor()(Image.open(img1))
        img2 = torchvision.transforms.ToTensor()(Image.open(img2))
        if(self.mode == "validation"):
            return torch.stack([img1, img2], dim=0), self.label_list[index]
        elif(self.mode == "test"):
            return torch.stack([img1, img2], dim=0)


# One block in my resnet, just the two-layer basic block in resnet with 3*3 filters
# used for implementing resnet18 or resnet34
class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        if stride == 1:
            self.shortcut = nn.Identity() if(out_channel == in_channel) else nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1)
        else:
            self.shortcut = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.conv_layers(x)
        shortcut = self.shortcut(x)
        out = self.relu(out + shortcut)
        return out


# resnet34
class ClassificationNetwork(nn.Module):
    # block_channels is a list of numbers of channels in each block
    # feature_dim is the dimension of my feature space
    def __init__(self, in_channel, num_classes, block_channels=[64, 128, 256, 512], feature_dim=512):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channel, block_channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(block_channels[0]),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),

            ResidualBlock(block_channels[0], block_channels[0]),
            ResidualBlock(block_channels[0], block_channels[0]),
            ResidualBlock(block_channels[0], block_channels[0]),

            ResidualBlock(block_channels[0], block_channels[1], stride=2),
            ResidualBlock(block_channels[1], block_channels[1]),
            ResidualBlock(block_channels[1], block_channels[1]),
            ResidualBlock(block_channels[1], block_channels[1]),

            ResidualBlock(block_channels[1], block_channels[2], stride=2),
            ResidualBlock(block_channels[2], block_channels[2]),
            ResidualBlock(block_channels[2], block_channels[2]),
            ResidualBlock(block_channels[2], block_channels[2]),
            ResidualBlock(block_channels[2], block_channels[2]),
            ResidualBlock(block_channels[2], block_channels[2]),

            ResidualBlock(block_channels[2], block_channels[3], stride=2),
            ResidualBlock(block_channels[3], block_channels[3]),
            ResidualBlock(block_channels[3], block_channels[3]),
            
            nn.AdaptiveAvgPool2d((1, 1)),   # For each channel, collapses (averages) the entire feature map (height & width) to 1x1
            nn.Flatten(),           # the above ends up with batch_size x block_channels[-1] x 1 x 1, flatten to batch_size x block_channels[-1]
        )
        # use features from self.layers, get logits
        self.linear_output = nn.Linear(block_channels[-1], num_classes)
    
    def forward(self, x, return_embedding=False):
        embedding = self.layers(x) 
        output = self.linear_output(embedding)
        if return_embedding:
            return embedding, output
        else:
            return output


class CenterLoss(nn.Module):
    """
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes, feat_dim, device=torch.device('cpu')):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12) # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()

        return loss


# train one epoch, return the average training loss and accuracy
def train_epoch(model, train_loader, criterion, optimizer):
    training_loss = 0
    training_acc = 0
    model.train()
    for batch_num, (input, rawlabel) in tqdm(enumerate(train_loader)):
        input = input.to(device)
        label = torch.tensor([train_index_to_class[int(i)] for i in rawlabel]).to(device)
        # train model
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        # get loss and accuracy
        training_loss += loss.item()
        _, predicted = torch.max(output, 1)
        training_acc += ((predicted == label).sum().item()/train_loader.batch_size)
    training_loss /= len(train_loader)
    training_acc /= len(train_loader)
    return training_loss, training_acc

# validation of classification task, return the average loss and accuracy
def evaluate_classification(model, val_loader, criterion):
    val_loss = 0
    val_acc = 0
    model.eval()
    with torch.no_grad():
        for batch_num, (input, rawlabel) in enumerate(val_loader):
            input = input.to(device)
            label = torch.tensor([train_index_to_class[int(i)] for i in rawlabel]).to(device)

            output = model(input)
            loss = criterion(output, label)

            val_loss += loss.item()
            _, predicted = torch.max(output, 1)
            val_acc += ((predicted == label).sum().item()/val_loader.batch_size)
    val_loss /= len(val_loader)
    val_acc /= len(val_loader)
    return val_loss, val_acc

# validation of verification task, return AUC score
def evaluate_verification(model, val_loader_verification, sim_metric):
    similarities = []
    labels = []
    model.eval()
    with torch.no_grad():
        for batch_num, (input, label) in enumerate(val_loader_verification):
            input1 = input[:,0].to(device)
            input2 = input[:,1].to(device)
            label = label.to(device)

            feature1, output1 = model(input1, return_embedding=True)
            feature2, output2 = model(input2, return_embedding=True)
            similarity = sim_metric(feature1, feature2)
            similarities.append(similarity.detach().cpu().numpy())
            labels.append(label.detach().cpu().numpy())
    val_auc_score = roc_auc_score(np.concatenate(labels, axis=0), np.concatenate(similarities, axis=0))
    return val_auc_score

def test_classification(model, test_loader):
    prediction = []
    model.eval()
    with torch.no_grad():
        for input in test_loader:
            input = input.to(device)
            output = model(input)
            _, predicted = torch.max(output, 1)
            prediction.append(predicted.detach().cpu().numpy())
    return np.concatenate(prediction, axis=0)

# return an array of cosine similarities of every pairs
def test_verification(model, test_loader_verification, sim_metric):
    similarities = []
    model.eval()
    with torch.no_grad():
        for input in test_loader_verification:
            input1 = input[:,0].to(device)
            input2 = input[:,1].to(device)
            feature1, output1 = model(input1, return_embedding=True)
            feature2, output2 = model(input2, return_embedding=True)
            similarity = sim_metric(feature1, feature2)
            similarities.append(similarity.detach().cpu().numpy())
    return np.concatenate(similarities, axis=0)


# center loss based training procedure

# train one epoch using center loss
def train_epoch_closs(model, train_loader, criterion_label, criterion_closs, optimizer_label, optimizer_closs, closs_weight):
    training_loss = 0
    training_acc = 0
    model.train()
    for batch_num, (input, rawlabel) in tqdm(enumerate(train_loader)):
        input = input.to(device)
        label = torch.tensor([train_index_to_class[int(i)] for i in rawlabel]).to(device)
        # train model
        optimizer_label.zero_grad()
        optimizer_closs.zero_grad()
        feature, output = model(input, return_embedding=True)
        loss_label = criterion_label(output, label)
        loss_center = criterion_closs(feature, label)
        loss = loss_label + closs_weight*loss_center
        loss.backward()
        optimizer_label.step()
        # by doing so, closs_weight would not impact on the learning of centers
        for param in criterion_closs.parameters():
            param.grad.data *= (1. / closs_weight)
        optimizer_closs.step()
        # get loss and accuracy
        training_loss += loss.item()
        _, predicted = torch.max(output, 1)
        training_acc += ((predicted == label).sum().item()/train_loader.batch_size)
        if(batch_num>100):break
    training_loss /= len(train_loader)
    training_acc /= len(train_loader)
    return training_loss, training_acc


# validation of classification task with center loss, return average loss and accuracy
def evaluate_classification_closs(model, val_loader, criterion_label, criterion_closs, closs_weight):
    val_loss = 0
    val_acc = 0
    model.eval()
    with torch.no_grad():
        for batch_num, (input, rawlabel) in enumerate(val_loader):
            input = input.to(device)
            label = torch.tensor([train_index_to_class[int(i)] for i in rawlabel]).to(device)

            feature, output = model(input, return_embedding=True)
            loss_label = criterion_label(output, label)
            loss_center = criterion_closs(feature, label)
            loss = loss_label + closs_weight*loss_center
            val_loss += loss.item()

            _, predicted = torch.max(output, 1)
            val_acc += ((predicted == label).sum().item()/val_loader.batch_size)
    val_loss /= len(val_loader)
    val_acc /= len(val_loader)
    return val_loss, val_acc


# validation of verification task with center loss, return AUC score
def evaluate_verification_closs(model, val_loader_verification, sim_metric):
    similarities = []
    labels = []
    model.eval()
    with torch.no_grad():
        for batch_num, (input, label) in enumerate(val_loader_verification):
            input1 = input[:,0].to(device)
            input2 = input[:,1].to(device)
            label = label.to(device)

            feature1, output1 = model(input1, return_embedding=True)
            feature2, output2 = model(input2, return_embedding=True)
            similarity = sim_metric(feature1, feature2)
            similarities.append(similarity.detach().cpu().numpy())
            labels.append(label.detach().cpu().numpy())
    try:
        val_auc_score = roc_auc_score(np.concatenate(similarities, axis=0), np.concatenate(labels, axis=0))
        return val_auc_score
    except ValueError:
        print(f"feature1 {feature1}, feature2: {feature2}, similarity: {similarities[-1]}")
        return 0

def test_classification_closs(model, test_loader):
  prediction = []
  model.eval()
  with torch.no_grad():
        for input in test_loader:
            input = input.to(device)
            output = model(input)
            _, predicted = torch.max(output, 1)
            prediction.append(predicted.detach().cpu().numpy())
  return np.concatenate(prediction, axis=0)

# return an array of cosine similarities of every pairs
def test_verification_closs(model, test_loader_verification, sim_metric):
  similarities = []
  model.eval()
  with torch.no_grad():
    for input in test_loader_verification:
      input1 = input[:,0].to(device)
      input2 = input[:,1].to(device)
      feature1, output1 = model(input1, return_embedding=True)
      feature2, output2 = model(input2, return_embedding=True)
      similarity = sim_metric(feature1, feature2)
      similarities.append(similarity.detach().cpu().numpy())
  return np.concatenate(similarities, axis=0)


if(__name__=="__main__"):
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    num_workers = 8 if cuda else 0

    # load data
    # the file need to be put under the folder where classification and verification data is unzipped
    batch_size = 64
    train_dataset = torchvision.datasets.ImageFolder(root='train_data', transform=torchvision.transforms.ToTensor())
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    train_dataset_DA = torchvision.datasets.ImageFolder(root='train_data', transform=torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomAffine(45, translate=(0.2, 0.2)),
        torchvision.transforms.ColorJitter(brightness=1, contrast=1, saturation=0.5, hue=0.5),
        torchvision.transforms.ToTensor()
    ]))
    train_dataloader_DA = DataLoader(train_dataset_DA, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_dataset = torchvision.datasets.ImageFolder(root='val_data', transform=torchvision.transforms.ToTensor())
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

    train_index_to_class = {i:int(c) for c, i in train_dataset.class_to_idx.items()}
    val_index_to_class = {i:int(c) for c, i in val_dataset.class_to_idx.items()}
    assert(train_index_to_class == val_index_to_class)

    val_dataset_verification = VerificationDataset("verification_data", "verification_pairs_val.txt", mode="validation")
    val_dataloader_verification = DataLoader(val_dataset_verification, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

    # create model and other stuff
    in_channel = 3 # RGB channels
    learningRate = 0.1
    lr_center = 0.1
    weightDecay = 5e-5
    num_classes = len(train_dataset.classes)
    block_channels = [64, 128, 256, 512]
    feature_dim = block_channels[-1]
    model = ClassificationNetwork(in_channel, num_classes, block_channels, feature_dim)
    model.to(device)
    closs_weight = 1
    criterion_label = nn.CrossEntropyLoss()
    # criterion_closs = CenterLoss(num_classes, feature_dim, device)
    optimizer_label = optim.SGD(model.parameters(), lr=learningRate, weight_decay=weightDecay, momentum=0.5)
    # optimizer_closs = optim.SGD(criterion_closs.parameters(), lr=lr_center)
    sim_metric = nn.CosineSimilarity(dim=1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_label, mode="max", factor=0.5, patience=2, threshold=0.005)
    # scheduler_closs = optim.lr_scheduler.ReduceLROnPlateau(optimizer_closs, mode="max", factor=0.5, patience=2, threshold=0.005)
    current_epoch = -1
    
    checkpoint = torch.load("best_checkpoint_final")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer_label.load_state_dict(checkpoint["optimizer_state_dict"])
    current_epoch = checkpoint["epoch"]
    criterion_label = checkpoint["loss"]
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    

    # train model
    epoch_num = 25
    loss_train = []
    loss_val = []
    acc_train = []
    acc_val = []
    best_val_acc = 0
    for epoch in range(current_epoch+1, epoch_num):
        # training_loss, training_acc = train_epoch_closs(model, train_dataloader, criterion_label, criterion_closs, optimizer_label, optimizer_closs, closs_weight)
        # val_loss, val_acc = evaluate_classification_closs(model, val_dataloader, criterion_label, criterion_closs, closs_weight)
        # val_auc_score = evaluate_verification_closs(model, val_dataloader_verification, sim_metric)
        training_loss, training_acc = train_epoch(model, train_dataloader_DA, criterion_label, optimizer_label)
        val_loss, val_acc = evaluate_classification(model, val_dataloader, criterion_label)
        val_auc_score = evaluate_verification(model, val_dataloader_verification, sim_metric)
        scheduler.step(val_acc)
        # scheduler_closs.step(val_acc)
        if(val_acc > best_val_acc):
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer_label.state_dict(),
                "loss": criterion_label,
                "scheduler_state_dict": scheduler.state_dict()
            }, "best_checkpoint_final")
        print(f"Epoch: {epoch}, training loss: {training_loss}, training accuracy: {training_acc}, validation loss: {val_loss}, validation accuracy: {val_acc}, validation AUC score: {val_auc_score}")
        if(epoch % 10 == 9):
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer_label.state_dict(),
                "loss": criterion_label,
                "scheduler_state_dict": scheduler.state_dict()
            }, f"best_checkpoint_new_{epoch}")
        loss_train.append(training_loss)
        loss_val.append(val_loss)
        acc_train.append(training_acc)
        acc_val.append(val_acc)

    # final test
    test_dataset = ClassificationTestDataset('test_data', "classification_test.txt")
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    test_dataset_verification = VerificationDataset("verification_data", "verification_pairs_test.txt", mode="test")
    test_dataloader_verification = DataLoader(test_dataset_verification, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    checkpoint = torch.load("best_checkpoint_final")
    model.load_state_dict(checkpoint["model_state_dict"])
    prediction = test_classification(model, test_dataloader)
    with open("classification_test.txt") as f:
        classification_test_files = f.readlines()
    with open("test_result_classification_new.csv", 'w') as f:
        f.write("id,label\n")
        for file, pred in zip(classification_test_files, prediction):
            f.write(f"{file.rstrip()},{pred}\n")
    similarities = test_verification(model, test_dataloader_verification, sim_metric)
    with open("verification_pairs_test.txt") as f:
        verification_test_files = f.readlines()
    with open("test_result_verification_new.csv", 'w') as f:
        f.write("Id,Category\n")
        for file, sim in zip(verification_test_files, similarities):
            f.write(f"{file.rstrip()},{sim}\n")