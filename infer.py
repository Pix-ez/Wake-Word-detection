"""the interface to interact with wakeword model"""
import pyaudio
import threading
import time
import argparse
import wave
import torchaudio
import torch
import numpy as np
from pydub import AudioSegment
from pydub.playback import play


# from neuralnet.dataset import get_featurizer
from threading import Event
from rcnn import Tinyrcnn
from custom_dataset import WakeWordDataset


class Listener:

    def __init__(self, sample_rate, record_seconds):
        self.chunk = 1024
        self.sample_rate = sample_rate
        self.record_seconds = record_seconds
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=self.sample_rate,
                        input=True,
                        output=True,
                        frames_per_buffer=self.chunk)

    def listen(self, queue):
        while True:
            data = self.stream.read(self.chunk , exception_on_overflow=False)
            queue.append(data)
            time.sleep(0.01)

    def run(self, queue):
        thread = threading.Thread(target=self.listen, args=(queue,), daemon=True)
        thread.start()
        print("\nWake Word Engine is now listening... \n")

# class Listener:
#     def __init__(self, sample_rate,  threshold):
#         self.buffer_size = 1024
#         self.sample_rate = sample_rate
#         self.p = pyaudio.PyAudio()
#         self.stream = self.p.open(format=pyaudio.paInt16,
#                         channels=1,
#                         rate=self.sample_rate,
#                         input=True,
#                         output=True,
#                         frames_per_buffer=self.buffer_size)
#         self.buffer = []
#         self.threshold = threshold

#     def listen(self, queue):
#         while True:
#             data = self.stream.read(self.buffer_size, exception_on_overflow=False)
#             self.buffer.append(data)
#             if len(self.buffer) >= self.buffer_size:
#                 # Apply noise filtering here if desired
#                 queue.append(self.buffer)  # Add the buffer to the queue
#                 self.buffer = self.buffer[-self.buffer_size:]  # Maintain buffer size
#             time.sleep(0.01)

#     def run(self, queue):
#         thread = threading.Thread(target=self.listen, args=(queue,), daemon=True)
#         thread.start()
#         print("\nWake Word Engine is now listening... \n")


class WakeWordEngine:

    def __init__(self, device , model_path, transform, sample_rate):
        self.listener = Listener(sample_rate=sample_rate, record_seconds=2)
        # self.listener = Listener(sample_rate=sample_rate, threshold=2)
        self.sample_rate= sample_rate
        self.device = device
        self.target_sample_rate = sample_rate
        self.num_samples = sample_rate
        # self.transform = transform
        self.transform = transform.to(self.device)
        self.model = Tinyrcnn(self.device)
        self.model.load_state_dict(torch.load(f=model_path))
        self.model.eval().to(self.device)

        # self.convert_mono = WakeWordDataset._mix_down_if_necessary()
        
        self.audio_q = list()

    def save(self, waveforms, fname="wakeword_temp"):
        wf = wave.open(fname, "wb")
        # set the channels
        wf.setnchannels(1)
        # set the sample format
        wf.setsampwidth(self.listener.p.get_sample_size(pyaudio.paInt16))
        # set the sample rate
        wf.setframerate(self.sample_rate)
        # write the frames as bytes
        wf.writeframes(b"".join(waveforms))
        # close the file
        wf.close()
        return fname
    
    def _mix_down_if_necessary(self, signal): 
        if signal.shape[0]> 1:
            signal = torch.mean(signal , dim=0 , keepdim=True)
        return signal
    
    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).to(self.device)
            signal = resampler(signal)
        return signal
    
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
    




    def predict(self, audio):
        with torch.no_grad():
            fname = self.save(audio)
            signal, sr = torchaudio.load(fname)
            signal = signal.to(self.device)
            signal = self._resample_if_necessary(signal, sr)
            signal = self._mix_down_if_necessary(signal)
            signal = self._cut_if_necessary(signal)
            signal = self._right_pad_if_necessary(signal)
            signal = self.transform(signal)
            signal = signal.unsqueeze(1)
            # print(f'Spectograph Shape{signal.shape}')


            # TODO: read from buffer instead of saving and loading file
            # waveform = torch.Tensor([np.frombuffer(a, dtype=np.int16) for a in audio]).flatten()
            # mfcc = self.featurizer(waveform).transpose(0, 1).unsqueeze(1)

            out = self.model(signal)
            pred = torch.round(torch.sigmoid(out))
            return pred.item()

    def inference_loop(self, action):
        while True:
            if len(self.audio_q) > 20:  # remove part of stream
                diff = len(self.audio_q) - 20
                for _ in range(diff):
                    self.audio_q.pop(0)
                action(self.predict(self.audio_q))
            elif len(self.audio_q) == 20:
                action(self.predict(self.audio_q))
            time.sleep(0.05)

    def run(self, action):
        self.listener.run(self.audio_q)
        thread = threading.Thread(target=self.inference_loop,
                                    args=(action,), daemon=True)
        thread.start()


class DemoAction:
    """This demo action will just randomly say Arnold Schwarzenegger quotes

        args: sensitivty. the lower the number the more sensitive the
        wakeword is to activation.
    """
    def __init__(self, sensitivity):
        # import stuff here to prevent engine.py from 
        # importing unecessary modules during production usage
        import os
        import subprocess
        import random
        from os.path import join, realpath
        import random
      
       

        self.random = random

      
        self.subprocess = subprocess
        self.detect_in_row = 0

        self.sensitivity = sensitivity
       
        folder = 'raju'
        self.raju_mp3 = [
            os.path.join(folder, x)
            for x in os.listdir(folder)
            if ".mp3" in x
        ]

    def __call__(self, prediction):
        if prediction == 1:
            self.detect_in_row += 1
            if self.detect_in_row == self.sensitivity:
                print("detected")
             
                self.play()
                self.detect_in_row = 0
        else:
            self.detect_in_row = 0

    def play(self):
        print("hello Rahul")
        filename = self.random.choice(self.raju_mp3)
        try:
            print("playing", filename)
            # Load the audio file using pydub
            audio = AudioSegment.from_file(filename)

                # Adjust the volume if needed (0.1 is 10% of the original volume)
            audio = audio - 10

            play(audio)
        except Exception as e:
            print(str(e))

    # def play(self):
    #     print("hello Rahul")
    #     filename = self.random.choice(self.raju_mp3)
    #     try:
    #         print("playing", filename)
    #         self.subprocess.check_output(['play', '-v', '.1', filename])
    #     except Exception as e:
    #         print(str(e))


if __name__ == "__main__":

    # parser = argparse.ArgumentParser(description="demoing the wakeword engine")
    # parser.add_argument('--model_file', type=str, default=None, required=True,
    #                     help='optimized file to load. use optimize_graph.py')
    # parser.add_argument('--sensitivty', type=int, default=10, required=False,
    #                     help='lower value is more sensitive to activations')


    # args = parser.parse_args()
    SAMPLE_RATE = 30000

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_SAVE_PATH = 'models\mel-sample-30k_epochs-100__lr-0.001.pth'

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


   

    wakeword_engine = WakeWordEngine(device=device, model_path=MODEL_SAVE_PATH, transform=mel_spectrogram, sample_rate=SAMPLE_RATE)

    action = DemoAction(sensitivity=4)
    # action = 'print("hello")'

    print("""\n*** Make sure you have sox installed on your system for the demo to work!!!
    If you don't want to use sox, change the play function in the DemoAction class
    in engine.py module to something that works with your system.\n
    """)
    # action = lambda x: print("hello")


 


  
 

    wakeword_engine.run(action)
    threading.Event().wait()