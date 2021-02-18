
## PRINT LAYER

class Print(nn.Module):
  def __init__(self):
    super(Print, self).__init__()

  def forward(self, x):
    print(x)
    return x

### RESHAPE LAYER

class View(nn.Module):
  # Implements reshaping capability as a distinct layer
  def __init__(self, shape):
    super(View, self).__init__()
    self.shape = shape

  def forward(self, x):
    batch_size = x.shape[0]
    return x.contiguous().view(batch_size, *self.shape)

### TIME-DISTRIBUTED LAYER

class TimeDistributed(nn.Module):

  def __init__(self, module):
    super(TimeDistributed, self).__init__()
    self.module = module

  def forward(self, x):
    ''' x size: (batch_size, measures, pitches * steps) '''
    batch_size, measures, p_and_s = x.size()
    input_tensor = x.contiguous().view(batch_size * measures, p_and_s)
    out = self.module(input_tensor)

    return out.contiguous().view(batch_size, measures, -1)

###

class autoencoder(nn.Module):
  def __init__(self):
    super(autoencoder, self).__init__()
    self.encoder = nn.Sequential(
      
      ### Level 1 encoding: encoding individual matrices
      View([MEASURES_PER_SAMPLE, NUMBER_OF_PITCHES * STEPS_PER_MEASURE]),
      # return (m, 16, 2000)
      TimeDistributed(nn.Linear(STEPS_PER_MEASURE * NUMBER_OF_PITCHES, 2000)),
      nn.ReLU(),
      # sqaush into (16m, 200), return (m, 16, 200)
      TimeDistributed(nn.Linear(2000, 200)),
      View([16*200]),
      nn.ReLU(),

      ### you'll have to reshape from (m, 16, 200) to (m, 3200) for the 
      # linear pass
      
      ### Level 2, linear passes
      nn.Linear(3200, 1600),
      nn.Linear(1600, 120),
      nn.BatchNorm1d(120, eps=1e-02)
      )

    self.decoder = nn.Sequential(

      nn.Linear(120, 1600),
      # Print(),
      nn.BatchNorm1d(1600, eps=1e-02),
      nn.ReLU(),
      nn.Dropout(p=0.1),

      nn.Linear(1600, 3200),
      View([MEASURES_PER_SAMPLE, 200]),
      TimeDistributed(nn.BatchNorm1d(200, eps=1e-02)),
      nn.ReLU(),

      TimeDistributed(nn.Linear(200, NUMBER_OF_PITCHES * STEPS_PER_MEASURE)),
      nn.Sigmoid(),
      View([MEASURES_PER_SAMPLE, NUMBER_OF_PITCHES, STEPS_PER_MEASURE])
      )

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x