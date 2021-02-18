# -*- coding: utf-8 -*-
"""Midi-Autoencoder.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LgwALMUORjmNUkCBPhaovtGXG0G4EO-e
"""

RUNNING_IN_GOOGLE_COLAB = None

try:
  if 'google.colab' in str(get_ipython()):
    RUNNING_IN_GOOGLE_COLAB = True
  else:
    RUNNING_IN_GOOGLE_COLAB = False
except NameError:
	RUNNING_IN_GOOGLE_COLAB = False


print(RUNNING_IN_GOOGLE_COLAB)

# LIBRARIES

# Native
import os

# 3rd Party
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable

# if not RUNNING_IN_GOOGLE_COLAB:
# 	from midi_processor import matrix_to_mid

# My Libraries
from network import autoencoder
# from midi_processor import matrix_to_mid

### FEATURE SIZES

MEASURES_PER_SAMPLE = 16
STEPS_PER_MEASURE = 96
NUMBER_OF_PITCHES = 96

### HYPERPARAMETERS

learning_rate = 0.001
NUM_EPOCHS = 2000
BATCH_SIZE = 256

CD_PATH = None
PATH_TO_DATA = None

if RUNNING_IN_GOOGLE_COLAB:
	
	CD_PATH = '/content/gdrive/MyDrive/Colab Notebooks/Midi-Autoencoder'

	from google.colab import drive

	drive.mount('/content/gdrive')
	PATH = '/content/gdrive/MyDrive/midi_matrices/midimatrices.npy'

else:

	## Here's where I'll have to specify the path to the data
	CD_PATH = os.path.join(os.path.abspath(os.getcwd()))

#ACCESS DATA SAVED ON DRIVE

np_data = np.load(PATH_TO_DATA)

dataloader = DataLoader(torch.from_numpy(np_data), 
                        batch_size=BATCH_SIZE, shuffle=True)

print(f"Split data into {len(dataloader)} batches of {BATCH_SIZE}")


### COMPILING MODEL

model = autoencoder()
criterion = nn.BCELoss()
optimizer = torch.optim.RMSprop(
    model.parameters(), lr=learning_rate, eps=1e-07)

def generate_random_song(epoch):
  """
  Pass a 120 unit 1d tensor through the decoder and generate a 
  song with the parameters as they are.
  """
  model.eval()
  with torch.no_grad():  
    sample_song = model.decoder(torch.rand(1, 120))
    assert(sample_song[0].shape == 
            (MEASURES_PER_SAMPLE, NUMBER_OF_PITCHES, STEPS_PER_MEASURE))
    
    song_np = ((sample_song[0].numpy())*127)
    song_int = song_np.astype(int)

    if not os.path.exists(f'{CD_PATH}/generated_songs/'):
      os.mkdir(f'{CD_PATH}/generated_songs/')

    np.save(
        f'{CD_PATH}/generated_songs/epoch{epoch}.npy', song_int)
  
  model.train()

generate_random_song(1) 

# print(model.decoder)

### Training

def train():

	print(f"Training {len(dataloader)} batches...")
	print()

	torch.autograd.set_detect_anomaly(True)

	for epoch in range(2000):
	  
	  running_loss = 0
	  
	  for data in dataloader:
	    # print(f"running batch {dataloader.index(data)}")
	    # I fucked up when I processed the data. I'm 
	    # squeezing the values between 0 and 1 here
	    song_batch = torch.where(data > 0, 1., 0.)
	    song_batch = Variable(song_batch)
	    # forward feed
	    output = model(song_batch)
	    loss = criterion(output, song_batch)
	    running_loss += loss.item()

	    # backward
	    optimizer.zero_grad()
	    loss.backward(retain_graph=True)
	    optimizer.step()
	  
	  # track results
	  print('epoch [{}/{}], loss: {:.4f}'
	          .format(epoch + 1, NUM_EPOCHS, running_loss))
	  
	  
	  
	  if epoch > 0 and epoch % 20 == 0:
	    
	    generate_random_song(epoch)

	    ### save model as history
	    if not os.path.exists(CD_PATH + '/History'):
	      os.mkdir(CD_PATH + '/History')
	    
	    torch.save(model.state_dict(), 
	                f'{CD_PATH}/History/model_e{epoch}.pt')

	    if not os.path.exists(CD_PATH + '/generated_songs'):
	      os.mkdir(CD_PATH + '/generated_songs')

	    generate_random_song(epoch)


if __name__ == '__main__':
	train()
