import os
import uuid
from pydub import AudioSegment
from pydub.silence import detect_leading_silence
from tqdm import tqdm

# Function to trim leading silence from an AudioSegment
trim_leading_silence = lambda x: x[detect_leading_silence(x):]

# Function to trim trailing silence from an AudioSegment
trim_trailing_silence = lambda x: trim_leading_silence(x.reverse()).reverse()

# Function to strip both leading and trailing silence from an AudioSegment
strip_silence = lambda x: trim_trailing_silence(trim_leading_silence(x))

# Specify the folder containing the WAV files
input_folder_path = "C:\\Users\\Rahul\\Documents\\Sound Recordings"

# Specify the folder where you want to save the trimmed audio with UUID filenames
output_folder_path = "D:\ml\wake\dataset\positive"

# Create the output folder if it doesn't exist
os.makedirs(output_folder_path, exist_ok=True)

# Get a list of WAV files in the input folder
wav_files = [filename for filename in os.listdir(input_folder_path) if filename.endswith(".wav")]

# Initialize the tqdm progress bar with green color and display the number of files
with tqdm(total=len(wav_files), ncols=100, bar_format="{l_bar}{bar:50}{r_bar}", colour='green') as pbar:
    for filename in wav_files:
        # Load the audio file
        sound = AudioSegment.from_file(os.path.join(input_folder_path, filename))
        
        # Apply the strip_silence function to trim silence
        stripped = strip_silence(sound)
        
        # Generate a unique UUID as the filename
        unique_filename = str(uuid.uuid4()) + ".wav"
        
        # Export the trimmed audio to the output folder with the UUID filename
        stripped.export(os.path.join(output_folder_path, unique_filename), format='wav')
        
        # Update the tqdm progress bar
        pbar.update(1)