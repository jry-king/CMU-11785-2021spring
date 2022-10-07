import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from tqdm import tqdm


# define dataset, used in train/validation/test
class MLPDataset(Dataset):

    def __init__(self, X, Y, context=20, feature_length=40):
        # directly assign the original data instead of copying to save memory
        # modification will also reflect to original data, so will be padded twice if created twice
        # only pad data on both ends
        # Y could be none in case of test
        self.length = len(X)
        self.context = context
        self.feature_length = feature_length
        self.X = torch.cat((torch.zeros((self.context, self.feature_length)), X, torch.zeros((self.context, self.feature_length))), dim=0)
        self.Y = Y
  

    def __len__(self):
        return self.length
  

    def __getitem__(self, index):
        x = self.X[index:index+2*self.context+1]
        # train/validation
        if(self.Y != None):
            y = self.Y[index]
            return x, y
        # test
        else:
            return x



# define model
class MLP(nn.Module):

    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),

            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),

            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),

            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),

            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),

            nn.Linear(512, 71)
        )
  

    def forward(self, x):
        return self.model(x)


# train one epoch, return the average training loss and accuracy
def train_epoch(model, train_loader, criterion, optimizer):
    training_loss = 0
    training_acc = 0
    model.train()
    for i, (input, label) in tqdm(enumerate(train_loader)):
        input = input.to(device)
        label = label.to(device)
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

# validation, return the average loss and accuracy
def evaluate(model, val_loader, criterion):
    val_loss = 0
    val_acc = 0
    model.eval()
    with torch.no_grad():
        for i, (input, label) in enumerate(val_loader):
            input = input.to(device)
            label = label.to(device)

            output = model(input)
            loss = criterion(output, label)

            val_loss += loss.item()
            _, predicted = torch.max(output, 1)
            val_acc += ((predicted == label).sum().item()/val_loader.batch_size)
    val_loss /= len(val_loader)
    val_acc /= len(val_loader)
    return val_loss, val_acc

def test(model, test_loader):
    prediction = []
    model.eval()
    with torch.no_grad():
        for input in test_loader:
            input = input.to(device)
            output = model(input)
            _, predicted = torch.max(output, 1)
            prediction.append(predicted.cpu().detach().numpy())
    return np.concatenate(prediction, axis=0)


if(__name__=="__main__"):
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    num_workers = 8 if cuda else 0

    # load data
    # directly stack all frames together to save memory by decrease the number of padding
    # use float32 to fit the model
    context = 25
    feature_length = 40
    train_X = torch.from_numpy(np.concatenate(np.load("train.npy", allow_pickle=True), axis=0)).float()
    train_Y = torch.from_numpy(np.concatenate(np.load("train_labels.npy", allow_pickle=True), axis=0)).long()
    train_dataset = MLPDataset(train_X, train_Y, context, feature_length)
    val_X = torch.from_numpy(np.concatenate(np.load("dev.npy", allow_pickle=True), axis=0)).float()
    val_Y = torch.from_numpy(np.concatenate(np.load("dev_labels.npy", allow_pickle=True), axis=0)).long()
    val_dataset = MLPDataset(val_X, val_Y, context, feature_length)
    # set drop_last=True for counting samples in calculating average accuracy
    batch_size = 1024
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # create model and other stuff
    sample_size = (context*2+1) * feature_length
    model = MLP(sample_size)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.5, step_size=2)

    # train model
    epoch_num = 10
    loss_train = []
    loss_val = []
    acc_train = []
    acc_val = []
    best_val_acc = 0
    for epoch in range(epoch_num):
        training_loss, training_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        scheduler.step()
        if(val_acc > best_val_acc):
            best_val_acc = val_acc
            torch.save(model.state_dict(), "bestmodel")
        print(f"Epoch: {epoch}, training loss: {training_loss}, training accuracy: {training_acc}, validation loss: {val_loss}, validation accuracy: {val_acc}")
        loss_train.append(training_loss)
        loss_val.append(val_loss)
        acc_train.append(training_acc)
        acc_val.append(val_acc)
    '''
    plt.figure(1)
    plt.plot(loss_train, 'b', label="train")
    plt.plot(loss_val, 'g', label="val")
    plt.title("loss")
    plt.legend()
    plt.show()
    plt.figure(2)
    plt.plot(acc_train, 'b', label="train")
    plt.plot(acc_val, 'g', label="val")
    plt.title("accuracy")
    plt.legend()
    plt.show()
    '''
    # final test
    test_X = torch.from_numpy(np.concatenate(np.load("test.npy", allow_pickle=True), axis=0)).float()
    test_dataset = MLPDataset(test_X, None, context, feature_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)   # don't set drop_last=True

    model.load_state_dict(torch.load("bestmodel"))
    prediction = test(model, test_loader)
    with open("test_result.csv", 'w') as f:
        f.write("id,label\n")
        for i, p in enumerate(prediction):
            f.write(f"{i},{p}\n")