import torch
import torch.nn as nn
import torchvision.models as models
import VGG_FACE
from torch.autograd import Variable
import gc
import numpy as np



class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class FeatureMapToSequence(nn.Module):

    def __init__(self):
        super(FeatureMapToSequence, self).__init__()
        
        self.feat_extract = nn.Sequential(
            nn.Linear(128*4,256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Dropout(0.5),
           ) 
        
        
    def forward(self, inputs):
        #input (nbatch, 64, unknown, 4)
        #output (nbatch, unknown, 64)
        
        inputs = inputs.permute(0,2,1,3)
        #print(inputs.size(), "afeter permute")
        input_sizes = inputs.size()
        inputs = inputs.contiguous().view(input_sizes[0], input_sizes[1], input_sizes[2] * input_sizes[3])
        #print(inputs.size(), "after view")
        input_sizes = inputs.size() 
        seq_length = input_sizes[1]
        seq_window = 50
        if seq_length < seq_window:
            seq_window = seq_length
        #if eval:
            #seq_window = seq_length
            #rand_start = 0

        rest = seq_length-seq_window
        if rest > 0:
            #rand_start = np.random.randint(rest)
            rand_start = 0
        else:
            rand_start = 0

        features = []
        for s in range(seq_window):
            input_slice = inputs[:,s,:].contiguous().view(-1,4*128)
                
            slice_features = self.feat_extract(input_slice)
            #print(slice_features.size(), "slice_features")
            features.append(slice_features)
        
        ## concatenate in (batch (words), sequence, features)
        features = torch.stack(features, dim=1)
        
        return features

class AudioFB_sequence_model(nn.Module):

    def __init__(self, nhid, nlayers, audio_embedding_size=16, dropout=0.5):
        super(AudioFB_sequence_model, self).__init__()
        
        
        self.audio_embedding_size=audio_embedding_size
        
        self.conv_feat_extractor = nn.Sequential( # Sequential,
	nn.Conv2d(1,32,5),
	nn.ReLU(),
	nn.Conv2d(32,64,3),
	nn.ReLU(),
        nn.MaxPool2d(1, 2),
	nn.Conv2d(64,128,3),
	nn.ReLU(),
        nn.MaxPool2d(2, 2), #(depende_de_n_frames,4 features)
        )
        
        #tengo 64 * nframes * 4 features, pero yo necesito (nframes, 64*4)
        self.feat_extract = FeatureMapToSequence()

        
        self.feature_size = 64                    
        #self.rnn = nn.RNN(64, nhid, nlayers, dropout=dropout, batch_first = True, bidirectional = False)
        #self.rnn = nn.RNN(64, nhid, nlayers, batch_first = True, bidirectional = False)
        self.rnn = nn.LSTM(self.feature_size, nhid, nlayers, batch_first = True, bidirectional = False)
        self.classifier = nn.Linear(nhid,self.audio_embedding_size)

        
        self.rnn_nhid = nhid
        self.rnn_layers = nlayers



    def forward(self, inputs, hidden, eval=False):
        ## Input is a sequence of faces for each user. batch dimension is in words (words, sequence, channels, height, width)

        print(inputs.size())
        
        if eval == False:
            inputs = torch.autograd.Variable(inputs).cuda()
        else:
            inputs = torch.autograd.Variable(inputs, volatile=True).cuda()
        cnn_features = self.conv_feat_extractor(inputs)
        
        #print(cnn_features.size())
        features = self.feat_extract(cnn_features)
        #print(features.size())

        #print(features.size())

        ## feed to the RNN
        output, hidden = self.rnn(features, hidden)
        
        #output = self.classifier(output[:,-1,:])
        
        #progressive weights
        #n_outputs = output.size()[1]
        #myweights = np.linspace(0.0, 1.0, n_outputs) #give progressive weights, from zero at start to 1 at the end of the sequence
        #myweights = myweights / np.sum(myweights)
        #outputs = []
        #for o in range(n_outputs):
            #outputs.append( self.classifier(output[:,o,:])  * myweights[o])
        #output = torch.sum(torch.stack(outputs, dim=2), dim=2).view(-1,7)
        
        n_outputs = output.size()[1]
        outputs = []
        for o in range(n_outputs):
            outputs.append( self.classifier(output[:,o,:])  )
        output = torch.mean(torch.stack(outputs, dim=2), dim=2).view(-1, self.audio_embedding_size)
        
        
        
        #print("output", output.size())
        #gc.collect()
        return output
      
    def init_hidden(self, bsz):
        hidden = (Variable(torch.zeros(self.rnn_layers, bsz, self.rnn_nhid)),
                Variable(torch.zeros(self.rnn_layers, bsz, self.rnn_nhid)))
        #hidden = Variable(torch.zeros(self.rnn_layers, bsz, self.rnn_nhid))
        return hidden
      
        #weight = next(self.parameters()).data
        #if self.rnn_type == 'LSTM':
            #return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    #Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        #else:
            #return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())



def model(**kwargs):
    return Vgg_face_sequence_model()
