import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset ,DataLoader


class WakeWordDataset(Dataset):
    def __init__(self,
                 csv_file,
                 transform,
                 target_sample_rate,
                 num_samples,
                 device):
        self.annotations  = pd.read_csv(csv_file)
       
        self.transform = transform
        self.transform = transform.to(device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.device = device
      
    def __len__(self):
        return len(self.annotations)
    

   
    
    # def __getitem__(self, index):
    #     audio_sample_path = self._get_audio_sample_path(index)
    #     label = self._get_audio_sample_label(index)
    #     signal, sr = torchaudio.load(audio_sample_path)
    #     signal = signal.to(self.device)
    #     signal = self._resample_if_necessary(signal, sr)
    #     signal = self._mix_down_if_necessary(signal)
    #     signal = self._cut_if_necessary(signal)
    #     signal = self._right_pad_if_necessary(signal)
    #     signal = self.transform(signal)
    #     return signal,label

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transform(signal)
        return signal, label
        

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal
    
    #(1 ,2 ,3 , -> 0,0,0,..)
    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0 , num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal
    
    
    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).to(self.device)
            signal = resampler(signal)
        return signal
    

    #Stereo to mono
    def _mix_down_if_necessary(self, signal): 
        if signal.shape[0]> 1:
            signal = torch.mean(signal , dim=0 , keepdim=True)
        return signal
    
   
    def _get_audio_sample_path(self, index):
        return self.annotations.iloc[index,1]
    
    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index,2]
        
        
        

 


if __name__ == "__main__":
    csv_file = 'data.csv'
    # positive_folder ='D:\\ml\\wake\\dataset\\1'
    # negative_folder ='D:\\ml\\wake\\dataset\\0'
    SAMPLE_RATE = 30000
    NUM_SAMPLES = 30000


    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'


    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=100
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

    dataset = WakeWordDataset(csv_file,mfcc_transform,
                              SAMPLE_RATE,NUM_SAMPLES,
                              device )

    print(f'no. data{len(dataset)}')
    batch_size = 32  # Adjust as needed
    train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True )
    data_iterator = iter(train_dataloader)
    sample , lable = data_iterator

    print(sample.shape)

    


