import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torchaudio
from tqdm import tqdm
from ctcdecode import CTCBeamDecoder
from Levenshtein import distance

# define dataset
class RNNDataset(Dataset):

    def __init__(self, X, Y):
        # directly assign the original data instead of copying to save memory
        # Y could be none in case of test
        self.length = len(X)
        self.X = X  # (number of samples, number of time steps, feature_length), ndarray of objects (ndarrays)
        self.Y = Y  # (number of samples, number of phoneme labels), ndarray of objects (lists)
  

    def __len__(self):
        return self.length
  

    def __getitem__(self, index):
        x = self.X[index]
        # train/validation
        if self.Y is not None:
            y = self.Y[index]
            return x, y
        # test
        else:
            return x

    # used to retrieve lengths of sequences and pad sequences
    # then convert result to tensor
    def collate_fn(self, data):
        inputs = []
        labels = []
        input_lengths = torch.zeros(len(data)).long()
        label_lengths = torch.zeros(len(data)).long()
        # train/val
        if self.Y is not None:
            for i, (x, y) in enumerate(data):
                inputs.append(torch.tensor(x))
                labels.append(torch.tensor(y))
                input_lengths[i] = len(x)
                label_lengths[i] = len(y)
            # arrange padded inputs and labels according to the input format of nn.CTCLoss
            padded_inputs = rnn.pad_sequence(inputs, batch_first=False)  # tensor of longest_input_length*batch_size*feature_length
            # data augmentation
            if self.length > 10000:
                train_transforms = nn.Sequential(
                    torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
                    torchaudio.transforms.TimeMasking(time_mask_param=35)
                )
                padded_inputs = train_transforms(padded_inputs.permute(1, 2, 0))
                padded_inputs = padded_inputs.permute(2, 0, 1)
            padded_labels = rnn.pad_sequence(labels, batch_first=True)  # tensor of batch_size*longest_label_length
            return padded_inputs.float(), padded_labels.long(), input_lengths, label_lengths
        # test
        else:
            for i, x in enumerate(data):
                inputs.append(torch.tensor(x))
                input_lengths[i] = len(x)
            padded_inputs = rnn.pad_sequence(inputs, batch_first=False)  # tensor of longest_input_length*batch_size*feature_length
            return padded_inputs.float(), input_lengths


# CNN + biLSTM
class Phoneme_classifier(nn.Module):
    # block_channels is a list of numbers of channels in each block
    # feature_dim is the dimension of my feature space
    def __init__(self, feature_length, rnn_hidden_size, num_classes, cnn_hidden_size=[80, 160, 320]):
        super().__init__()
    
        self.cnn = nn.Sequential(
            nn.Conv1d(feature_length, cnn_hidden_size[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(cnn_hidden_size[0]),
            nn.ReLU(),
            nn.Conv1d(cnn_hidden_size[0], cnn_hidden_size[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(cnn_hidden_size[1]),
            nn.ReLU(),
            nn.Conv1d(cnn_hidden_size[1], cnn_hidden_size[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(cnn_hidden_size[2]),
            nn.ReLU(),
        )
        # set batch_first=False because of the requirement of nn.CTCLoss
        self.rnn = nn.LSTM(input_size=cnn_hidden_size[-1], hidden_size=rnn_hidden_size, num_layers=3, dropout=0.25, bidirectional=True)
        # longest_input_length*batch_size*feature_length -> longest_input_length*batch_size*hidden_size*2(bidirectional)
        self.out = nn.Sequential(
            nn.Linear(rnn_hidden_size*2, rnn_hidden_size*2),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(rnn_hidden_size*2, num_classes),
        )
        '''
        nn.Linear(rnn_hidden_size*2, rnn_hidden_size),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(rnn_hidden_size, num_classes),
        '''
        # longest_input_length*batch_size*hidden_size*2 -> longest_input_length*batch_size*num_classes
        # actually take first dimension as batch_size, but doesn't matter
  
    # input: N * time_steps * feature_length (padded)
    # output: N * time_steps * num_classes
    def forward(self, x, lengths):
        # change input from longest_input_length*batch_size*feature_length to batch_size*feature_length*longest_input_length for CNN
        # the result would be of shape batch_size*final_hidden*sequence_length (sequence_length is unchanged, which is just longest_input_length)
        x_encode = self.cnn(x.permute(1, 2, 0))
        # change shape back to longest_input_length*batch_size*feature_length for RNN
        # again, batch_first=False, for nn.CTCLoss
        packed_x = rnn.pack_padded_sequence(x_encode.permute(2, 0, 1), lengths.cpu(), enforce_sorted=False)
        packed_out = self.rnn(packed_x)[0]
        output, out_lengths = rnn.pad_packed_sequence(packed_out)
        # Log softmax after output layer is required since nn.CTCLoss expects log probabilities
        output = self.out(output).log_softmax(2)
        return output, out_lengths
"""
# pure rnn model, use biLSTM
class RNN_model(nn.Module):
  # block_channels is a list of numbers of channels in each block
  # feature_dim is the dimension of my feature space
  def __init__(self, feature_length, rnn_hidden_size, num_classes, cnn_hidden_size=[]):
    super().__init__()
    
    # set batch_first=False because of the requirement of nn.CTCLoss
    self.rnn = nn.LSTM(input_size=feature_length, hidden_size=rnn_hidden_size, num_layers=2, dropout=0.2, bidirectional=True)
    # longest_input_length*batch_size*feature_length -> longest_input_length*batch_size*hidden_size*2(bidirectional)
    self.out = nn.Linear(hidden_size*2, num_classes)
    # longest_input_length*batch_size*hidden_size*2 -> longest_input_length*batch_size*num_classes
    # actually take first dimension as batch_size, but doesn't matter
  
  # input: N * time_steps * feature_length (padded)
  # output: N * time_steps * num_classes
  def forward(self, x, lengths):
    # again, batch_first=False, for nn.CTCLoss
    packed_x = rnn.pack_padded_sequence(x, lengths.cpu(), enforce_sorted=False)
    packed_out = self.rnn(packed_x)[0]
    output, out_lengths = rnn.pad_packed_sequence(packed_out)
    # Log softmax after output layer is required since nn.CTCLoss expects log probabilities
    output = self.out(output).log_softmax(2)
    return output, out_lengths
"""


# train one epoch, return the average training loss and accuracy
def train_epoch(model, train_loader, criterion, optimizer):
    training_loss = 0
    model.train()
    for batch_num, (padded_inputs, padded_labels, input_lengths, label_lengths) in tqdm(enumerate(train_loader)):
        padded_inputs = padded_inputs.to(device)
        padded_labels = padded_labels.to(device)
        input_lengths = input_lengths.to(device)
        label_lengths = label_lengths.to(device)
        # train model
        optimizer.zero_grad()
        output, out_lengths = model(padded_inputs, input_lengths)
        loss = criterion(output, padded_labels, out_lengths, label_lengths)
        loss.backward()
        optimizer.step()
        # get loss and accuracy
        training_loss += loss.item()
    training_loss /= len(train_loader)
    return training_loss

# validation of classification task, return the average loss and accuracy
def evaluate(model, val_loader, criterion, decoder):
    val_loss = 0
    val_LD = 0
    model.eval()
    with torch.no_grad():
        for batch_num, (padded_inputs, padded_labels, input_lengths, label_lengths) in enumerate(val_loader):
            padded_inputs = padded_inputs.to(device)
            padded_labels = padded_labels.to(device)
            input_lengths = input_lengths.to(device)
            label_lengths = label_lengths.to(device)

            output, out_lengths = model(padded_inputs, input_lengths)
            loss = criterion(output, padded_labels, out_lengths, label_lengths)
            results, _, _, result_lengths = decoder.decode(output.transpose(0, 1), out_lengths) # different from ctcloss, need transpose
            # translate phoneme index sequences to character encoding sequences
            letter_seqs = []
            for i in range(len(results)):
                result = results[i][0][:result_lengths[i][0]]
                letter_seqs.append(''.join([PHONEME_MAP[j] for j in result]))
            letter_label_seqs = []
            for i in range(len(padded_labels)):
                label_seq = padded_labels[i][:label_lengths[i]]
                letter_label_seqs.append(''.join([PHONEME_MAP[j] for j in label_seq]))
            assert(len(results) == len(padded_labels))

            val_loss += loss.item()
            batch_LD = 0
            batch_size = len(padded_labels)
            for i in range(batch_size):
                batch_LD += distance(letter_seqs[i], letter_label_seqs[i])
            batch_LD /= batch_size
            val_LD += batch_LD
    val_loss /= len(val_loader)
    val_LD /= len(val_loader)
    return val_loss, val_LD

def test(model, test_loader, decoder):
    prediction = []
    model.eval()
    with torch.no_grad():
        for padded_inputs, input_lengths in test_loader:
            padded_inputs = padded_inputs.to(device)
            input_lengths = input_lengths.to(device)
            output, out_lengths = model(padded_inputs, input_lengths)
            results, _, _, result_lengths = decoder.decode(output.transpose(0, 1), out_lengths) # different from ctcloss, need transpose
            # translate phoneme index sequences to character encoding sequences
            for i in range(len(results)):
                result = results[i][0][:result_lengths[i][0]]
                prediction.append(''.join([PHONEME_MAP[j] for j in result]))
    return prediction


N_PHONEMES = 41
PHONEME_LIST = [
    " ",
    "SIL",
    "SPN",
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "ER",
    "EY",
    "F",
    "G",
    "H",
    "IH",
    "IY",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OY",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH",
    "UW",
    "V",
    "W",
    "Y",
    "Z",
    "ZH"
]

PHONEME_MAP = [
    " ",
    ".", #SIL
    "!", #SPN
    "a", #AA
    "A", #AE
    "h", #AH
    "o", #AO
    "w", #AW
    "y", #AY
    "b", #B
    "c", #CH
    "d", #D
    "D", #DH
    "e", #EH
    "r", #ER
    "E", #EY
    "f", #F
    "g", #G
    "H", #H
    "i", #IH 
    "I", #IY
    "j", #JH
    "k", #K
    "l", #L
    "m", #M
    "n", #N
    "N", #NG
    "O", #OW
    "Y", #OY
    "p", #P 
    "R", #R
    "s", #S
    "S", #SH
    "t", #T
    "T", #TH
    "u", #UH
    "U", #UW
    "v", #V
    "W", #W
    "?", #Y
    "z", #Z
    "Z" #ZH
]


cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
num_workers = 4 if cuda else 0

train_X = np.load("train.npy", allow_pickle=True)
train_Y = np.load("train_labels.npy", allow_pickle=True)
val_X = np.load("dev.npy", allow_pickle=True)
val_Y = np.load("dev_labels.npy", allow_pickle=True)
test_X = np.load("test.npy", allow_pickle=True)
# load training data
# set drop_last=True for counting samples in calculating average accuracy
batch_size = 32
train_dataset = RNNDataset(train_X, train_Y)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, collate_fn=train_dataset.collate_fn)
val_dataset = RNNDataset(val_X, val_Y)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers, collate_fn=val_dataset.collate_fn)
# test data
test_dataset = RNNDataset(test_X, None)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=test_dataset.collate_fn)

# create model and other stuff
feature_length = 40
learningRate = 2e-3
weightDecay = 5e-6
num_classes = N_PHONEMES + 1
hidden_size = 512
model = Phoneme_classifier(feature_length, hidden_size, num_classes)
model.to(device)

criterion = nn.CTCLoss()
optimizer = optim.Adam(model.parameters(), lr=learningRate, weight_decay=weightDecay)
# scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.5, step_size=5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, threshold=0.5)
decoder = CTCBeamDecoder(PHONEME_LIST, beam_width=6, log_probs_input=True)
'''
checkpoint = torch.load("best_checkpoint_DA_newlr")
model.load_state_dict(checkpoint["model_state_dict"])
# checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"] = 5e-4
criterion = checkpoint["loss"]
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
trained_epoch = checkpoint["epoch"] + 41
# scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, threshold=0.1)
'''
# train model
# train 40 epochs, then manually turn up lr, then train another 20 epochs
epoch_num = 40
best_val_LD = np.inf
for epoch in range(epoch_num):
    training_loss = train_epoch(model, train_dataloader, criterion, optimizer)
    val_loss, val_LD = evaluate(model, val_dataloader, criterion, decoder)
    scheduler.step(val_LD)
    if(val_LD < best_val_LD):
        best_val_LD = val_LD
        torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": criterion,
                "scheduler_state_dict": scheduler.state_dict()
            }, "best_checkpoint_DA_newlr")
    if(epoch % 10 == 9):
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": criterion,
            "scheduler_state_dict": scheduler.state_dict()
        }, f"checkpoint_{epoch}_DA_newlr")
    print(f"Epoch: {epoch}, training loss: {training_loss}, validation loss: {val_loss}, validation Levenshtein Distance: {val_LD}")

# test
checkpoint = torch.load("best_checkpoint_DA_newlr")
model.load_state_dict(checkpoint["model_state_dict"])
prediction = test(model, test_dataloader, decoder)
with open("test_result.csv", 'w') as f:
    f.write("id,label\n")
    for i, pred in enumerate(prediction):
        f.write(f"{i},{pred}\n")