import torch
import torch.nn as nn
import torchvision.models as models
import VGG_FACE
from torch.autograd import Variable
import gc
import numpy as np

class Word_Embeddings_sequence_model(nn.Module):

    def __init__(self, vocab_size, embedding_size, nhid, nlayers, dropout=0.5, pretrained_embedding_path = None):
        super(Word_Embeddings_sequence_model, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        
        self.rnn_nhid = nhid
        self.rnn_layers = nlayers
        self.bidirectional = False
        self.seq_window = 100

        self.rnn = nn.LSTM(embedding_size, nhid, nlayers, dropout=dropout, batch_first = True, bidirectional = self.bidirectional)
        #self.rnn = nn.RNN(64, nhid, nlayers, batch_first = True, bidirectional = False)
        #self.rnn = nn.LSTM(64, nhid, nlayers, batch_first = True, bidirectional = False)

        #A different classifier for each personality traits, since they are not mutually exclusive
        self.classifiers = []
        for i in range(6):
            self.classifiers.append( nn.Linear(nhid,1))
    



    def forward(self, inputs, hidden, eval=False):
        ## Input is a sequence of faces for each user. batch dimension is in users (users, sequence, channels, height, width)
        
        #first get a slice for earch sequence element, get features from convolutional and store output in another sequence to feed the RNN
        input_sizes = inputs.size() 
        seq_length = input_sizes[1]
        seq_window = self.seq_window
        if seq_length < seq_window:
            seq_window = seq_length
        if eval:
            seq_window = seq_length
            #rand_start = 0

        rest = seq_length-seq_window
        if rest > 0:
            rand_start = np.random.randint(rest)
        else:
            rand_start = 0

        cropped_input = inputs.narrow(1,rand_start,seq_window).contiguous().view(input_sizes[0], seq_window)

        if eval == False:
            cropped_input = torch.autograd.Variable(cropped_input).cuda()
        else:
            cropped_input = torch.autograd.Variable(cropped_input, volatile=True).cuda()

        embeddings = self.embedding(cropped_input)

        ## feed to the RNN
        output, hidden = self.rnn(embeddings, hidden)
                
        n_outputs = output.size()[1]
        total_output = []
        for classifier in self.classifiers:
            outputs = []
            for o in range(n_outputs):
                outputs.append( classifier(output[:,o,:])  )
            output = torch.mean(torch.stack(outputs, dim=2), dim=2).view(-1,1)
            total_output.append(output)
        final_output = torch.stack(total_output, dim=2).view(-1,6)
        return final_output
      
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


