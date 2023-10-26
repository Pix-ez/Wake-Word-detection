﻿# Wake-Word-detection-

This is a Wake word detection system made with using RCNN model which is Convolutional network used with Recurrent neural network here used Gated recurrent units (GRUs).

This paper is used as reference for model architecture - [(https://arxiv.org/abs/2109.14725)]


### using virtualenv (recommend)
I am using windows.
1. `python -m virtualenv wakeword`
2. `wakeword/Scripts/activate`

### pip packages
`pip install -r requirements.txt` 

Dataset contains two type of data positive and negative with dir - dataset/0 , dataset/1 

positive data contains recording of saying wake word  and negative data is random data noise.

Infernce code is not perfect has some bugs , model is also not trained on lots of data , hence it gets activated with similar sounding words.

1. Collect data and run data_clean.py to clean it
2. Change custom_dataset.py as require.
3. Change rcnn.py model as require.
4. Train set hyperparameters inside the train.py file.
5. Test with infer.py .
