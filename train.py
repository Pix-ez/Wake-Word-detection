from custom_dataset import WakeWordDataset
from rcnn import Tinyrcnn
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn 
import torchaudio
from torchmetrics.classification import BinaryAccuracy
from torch.utils.data import Dataset , DataLoader , random_split
import subprocess



# Specify the log directory
log_dir = 'logs'

# Start TensorBoard
subprocess.Popen(['tensorboard', '--logdir', log_dir])


#tensorboard
writer_train = SummaryWriter(f"logs/train")
writer_test = SummaryWriter(f"logs/test")
step = 0

#import data and dataloader 
csv_file = 'data.csv'

SAMPLE_RATE = 30000
NUM_SAMPLES = 30000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=150
    )

n_fft = 2048
win_length = None
hop_length = 512
n_mels = 200
n_mfcc = 200

mfcc_transform = torchaudio.transforms.MFCC(
sample_rate=SAMPLE_RATE,
n_mfcc=n_mfcc,
melkwargs={
    "n_fft": n_fft,
    "n_mels": n_mels,
    "hop_length": hop_length,
    },
)

dataset = WakeWordDataset(csv_file,
                          mel_spectrogram,
                          SAMPLE_RATE,
                          NUM_SAMPLES,
                          device)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

BATCH_SIZE = 32
train_dataloader = DataLoader(train_dataset, BATCH_SIZE , shuffle=True)
test_dataloader = DataLoader(test_dataset, BATCH_SIZE , shuffle=True)

#setup model
model = Tinyrcnn(device).to(device)

# print(f"using device= {device}")
# print(model)

#setup hyperparamters
EPOCH = 100
LEARNING_RATE = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss()

def train(model, device, train_loader, optimizer, loss_fn, epoch):

    model.train()
    criterion = loss_fn
    metric = BinaryAccuracy().to(device)
    accuracy = 0
    

   

    for batch_idx, (sample, lable) in enumerate(train_loader):
        sample, lable = sample.to(device) , lable.to(device)
        # print(f'sample {sample.shape}')
        optimizer.zero_grad()
        output = model(sample)
        # print(output.shape)
        lable = lable.float().unsqueeze(1)
        # print(lable.dtype)
        loss = criterion(output, lable)
        accuracy += metric(output,lable)
        loss.backward()
        optimizer.step()

        # if batch_idx == 0:
        #     print(
        #         f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(train_loader)} \
        #               Loss D: {loss:.4f}"
        #     )
        

        if batch_idx % 100 ==0:
            print('Train Loss: {:.6f} \t train accuracy: {:.3f}'.format( loss.item(), accuracy))
            writer_train.add_scalar('Loss', loss, epoch)


def test(model, device, test_loader, loss_fn):
    test_loss = 0
    model.eval()
    criterion = loss_fn
    metric = BinaryAccuracy().to(device)
    accuracy = 0

    with torch.inference_mode():
        total_samples = 0  # To keep track of the total number of samples
 
        data_iterator = iter(test_loader)
        sample, lable = next(data_iterator)
        
        sample, lable = sample.to(device), lable.to(device)
        output = model(sample)
        lable = lable.float().unsqueeze(1)
        batch_size = sample.size(0)  # Get the actual batch size of the current batch
        test_loss += criterion(output, lable) * batch_size
        accuracy += metric(output,lable)
        total_samples += batch_size

        test_loss /= total_samples  # Calculate the average loss over all samples
        print(f"Test loss: {test_loss:.5f}\t test accuracy: {accuracy:.3f}\n")
        writer_test.add_scalar('Loss', test_loss, epoch)


for epoch in range(EPOCH):
    print('Epoch: {} '.format(epoch+1))
    train(model, device, train_dataloader, optimizer, criterion, EPOCH)
    test(model,device, test_dataloader,criterion)

    

MODEL_SAVE_PATH = f'models/mel_epochs-{epoch+1}__lr-{LEARNING_RATE}.pth'
torch.save(model.state_dict(), MODEL_SAVE_PATH)

# tensorboard --logdir=runs
