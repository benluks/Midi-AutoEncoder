import torch
from torch import nn

from midi_processor import MEASURES_PER_SAMPLE, STEPS_PER_MEASURE, NUMBER_OF_PITCHES


class View(nn.Module):
    def __init__(self, shape):
		super(View, self).__init__()
        self.shape = shape


    def forward(self, x):
    	batch_size = x.shape[0]
        return x.view(batch_size, *self.shape)


class TimeDistributed(nn.Module):

    def __init__(self, module, batch_first=False, linear_out=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        ''' x size: (batch_size, measures, pitches * steps) '''
        batch_size, measures, p_and_s = x.size()
        input_tensor = x.view(batch_size * measures, p_and_s)
        output = self.module(input_tensor)
        
        # Whether to return flat tensor for linear input or 2D tensor
        # (not counting batch_size) for input into another TS layer
        if linear_out:       	
        	return output.view(batch_size, -1)
        else: 	
        	return output.view(batch_size, measures, -1)



class autoencoder(nn.Module):
	def __init__(self):
		super(autoencoder, self).__init__()
		self.encoder = nn.Sequential(
			
			### Level 1 encoding: encoding individual matrices
			View([MEASURES_PER_SAMPLE, NUMBER_OF_PITCHES * STEPS_PER_MEASURE]),
			TimeDistributed(nn.Linear(STEPS_PER_MEASURE * NUMBER_OF_PITCHES, 2000)),	# return (m, 16, 2000)
			nn.ReLU(True),
			TimeDistributed(nn.Linear(2000, 200), linear_out=True),	# sqaush into (16m, 2000), return (m, 16, 200)
			nn.ReLU(True),
			
			### Level 2, linear passes
			nn.Linear(3200, 1600),
			nn.Linear(1600, 120),
			nn.BatchNorm1d(120, eps=1e-02, momentum=0.9)
			)

		self.decoder = nn.Sequential(

			nn.Linear(120, 1600),
			nn.BatchNorm1d(1600, eps=1e-02, momentum=0.9),
			nn.ReLU(True),
			nn.Dropout(p=0.1),

			nn.Linear(1600, 3200),
			View(16, 200),	# I may have to add a parameters for m
			TimeDistributed(nn.BatchNorm1d(200, eps=1e-02, momentum=0.9)),
			nn.ReLU(True),

			TimeDistributed(nn.Linear(200, NUMBER_OF_PITCHES * STEPS_PER_MEASURE)),
			nn.Sigmoid(),
			View([MEASURES_PER_SAMPLE, NUMBER_OF_PITCHES, STEPS_PER_MEASURE])
			)

	def forward(self, x):
		### reshape input before forward pass
		x = self.encoder(x)
		x = self.decoder(x)
		return x

model = autoencoder().cuda()
criterion = nn.BCELoss()
optimizer = torch.optim.RMSprop(lr=0.001, eps=1e-07)



