import torch
import torch.nn as nn
import torchvision.models as models
import VGG_FACE
from torch.autograd import Variable
import gc
import numpy as np

class Vgg_face_sequence_model(nn.Module):

    def __init__(self, nhid, nlayers, face_embedding_size=16, dropout=0.5, pretrained_model_path = None):
        super(Vgg_face_sequence_model, self).__init__()
        
        
        self.face_embedding_size=face_embedding_size
        #model = VGG_FACE.VGG_FACE
        #model.load_state_dict(torch.load('VGG_FACE.pth'))
        #for param in model.parameters():
            #param.requires_grad = False
        #list_model = list(model.children())
        #del list_model[-1] #delete softmax
        #del list_model[-1] #delete last_output_layer

        #model =  nn.Sequential(*list_model)
        #self.vgg_face = model
        #model = None
        #list_model = None
        #print(self.vgg_face)
        
        #if pretrained_model_path is not None:
            ##print("=> loading checkpoint '{}'".format(pretrained_model_path))
            #checkpoint = torch.load(pretrained_model_path)
            ##start_epoch = checkpoint['epoch']
            ##best_prec1 = checkpoint['best_prec1']
            #self.vgg_face.load_state_dict(checkpoint['state_dict'])
            #for param in self.vgg_face.parameters():
                #param.requires_grad = False
            #print(self.vgg_face)



        #model = self.vgg_face
        ##self.vgg_face = torch.nn.DataParallel(model).cuda()
        #self.vgg_face = model.cuda()
        #self.vgg_face.eval()
        ##model = None
        ##self.vgg_face.cuda()
        #print(self.vgg_face)

        self.bidirectional = True
        if self.bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        
        self.feature_size = 128                    
        #self.rnn = nn.RNN(64, nhid, nlayers, dropout=dropout, batch_first = True, bidirectional = False)
        #self.rnn = nn.RNN(64, nhid, nlayers, batch_first = True, bidirectional = False)
        self.rnn = nn.LSTM(self.feature_size, nhid, nlayers, batch_first = True, bidirectional = self.bidirectional, dropout=0.2)
        #self.classifier = nn.Linear(nhid  * self.num_directions, self.face_embedding_size)

        self.feat_extract = torch.nn.Sequential(
          nn.Linear(4096,self.feature_size),
          nn.ReLU(),
          nn.Dropout(0.5)
        )

        
        self.final_feature_size = 128
        self.rnn_features = nn.Linear(nhid * self.num_directions, self.final_feature_size)
        self.classifier = nn.Linear(self.final_feature_size,self.face_embedding_size)
        
        self.rnn_nhid = nhid
        self.rnn_layers = nlayers



    def forward(self, inputs, hidden, eval=False):
        ## Input is a sequence of faces for each user. batch dimension is in words (words, sequence, channels, height, width)
        
        #first get a slice for earch sequence element, get features from convolutional and store output in another sequence to feed the RNN
        
        #print(inputs.size())
        input_sizes = inputs.size() 
        seq_length = input_sizes[1]
        seq_window = 30
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


        #flat_raw_features = inputs.view(-1, input_sizes[1]*input_sizes[2])
        #if eval == False:
            #flat_raw_features = torch.autograd.Variable(flat_raw_features).cuda()
        #else:
            #flat_raw_features = torch.autograd.Variable(flat_raw_features, volatile=True).cuda()
        #print(flat_raw_features.size())
        #flat_features = self.feat_extract(flat_raw_features)
        #features = flat_features.view(-1, input_sizes[1], self.feature_size)
        #print(features.size())

        features = []
        for s in range(seq_window):
            input_slice = inputs[:,s,:].contiguous().view(-1,4096)
            if eval == False:
                input_slice = torch.autograd.Variable(input_slice).cuda()
            else:
                input_slice = torch.autograd.Variable(input_slice, volatile=True).cuda()
                
            slice_features = self.feat_extract(input_slice)
            features.append(slice_features)
            
        
        ## concatenate in (batch (words), sequence, features)
        features = torch.stack(features, dim=1)
        #print(features.size())

        ## feed to the RNN
        output, hidden = self.rnn(features, hidden)
        
        #output = self.classifier(output[:,-1,:])
        #return output
        
        #progressive weights
        #n_outputs = output.size()[1]
        #myweights = np.linspace(0.0, 1.0, n_outputs) #give progressive weights, from zero at start to 1 at the end of the sequence
        #myweights = myweights / np.sum(myweights)
        #outputs = []
        #for o in range(n_outputs):
            #outputs.append( self.classifier(output[:,o,:])  * myweights[o])
        #output = torch.sum(torch.stack(outputs, dim=2), dim=2).view(-1,7)
        
        #n_outputs = output.size()[1]
        #outputs = []
        #for o in range(n_outputs):
            #outputs.append( self.classifier(output[:,o,:])  )
        #output = torch.mean(torch.stack(outputs, dim=2), dim=2).view(-1, self.face_embedding_size)
        #return output
        
        n_outputs = output.size()[1]
        outputs = []
        for o in range(n_outputs):
            outputs.append( self.rnn_features(output[:,o,:])  )
        final_feature = torch.mean(torch.stack(outputs, dim=2), dim=2).view(-1,self.final_feature_size)

        final_output = self.classifier(final_feature)
        return final_output    
    
    
    def init_hidden(self, bsz):
        hidden = (Variable(torch.zeros(self.rnn_layers * self.num_directions, bsz, self.rnn_nhid)),
                Variable(torch.zeros(self.rnn_layers * self.num_directions, bsz, self.rnn_nhid)))
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
