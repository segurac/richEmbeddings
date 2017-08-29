import torch
import torch.nn as nn
import torchvision.models as models
import VGG_FACE
from torch.autograd import Variable
import gc
import numpy as np

class Vgg_face_sequence_model(nn.Module):

    def __init__(self, nhid, nlayers, dropout=0.5, pretrained_model_path = None):
        super(Vgg_face_sequence_model, self).__init__()
        
        model = VGG_FACE.VGG_FACE
        model.load_state_dict(torch.load('VGG_FACE.pth'))
        for param in model.parameters():
            param.requires_grad = False
        list_model = list(model.children())
        del list_model[-1] #delete softmax
        list_model[-1] =  torch.nn.Sequential(VGG_FACE.Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),torch.nn.Linear(4096,64))
        list_model.append(  nn.ReLU() )
        list_model.append(  nn.Dropout(0.5) )
        list_model.append( torch.nn.Sequential(VGG_FACE.Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),torch.nn.Linear(64,7)) )
        model =  nn.Sequential(*list_model)
        self.vgg_face = torch.nn.DataParallel(model)
        #self.vgg_face = model
        model = None
        list_model = None
        print(self.vgg_face)
        
        if pretrained_model_path is not None:
            #print("=> loading checkpoint '{}'".format(pretrained_model_path))
            checkpoint = torch.load(pretrained_model_path)
            #start_epoch = checkpoint['epoch']
            #best_prec1 = checkpoint['best_prec1']
            self.vgg_face.load_state_dict(checkpoint['state_dict'])
            for param in self.vgg_face.parameters():
                param.requires_grad = False
            print(self.vgg_face)


        ## remove last fully-connected layer
        #model = nn.Sequential(*list(self.vgg_face.children())[:-1])
        if pretrained_model_path is not None:
            model = nn.Sequential(*list(next(self.vgg_face.children()).children())[:-1])
        else:
            model = nn.Sequential(*list(self.vgg_face.children())[:-1])

        self.vgg_face = torch.nn.DataParallel(model).cuda()
        #self.vgg_face = model
        model = None
        #self.vgg_face.cuda()
        print(self.vgg_face)

        ## Do not train vgg16 parameters
        #for m in self.vgg_face_sequence_model.classifier.children():
            #print(m)
            #if isinstance(m,torch.nn.Linear):
                #for p in m.parameters():
                    #p.requires_grad=False
                    
        #self.rnn = nn.RNN(64, nhid, nlayers, dropout=dropout, batch_first = True, bidirectional = False)
        #self.rnn = nn.RNN(64, nhid, nlayers, batch_first = True, bidirectional = False)
        self.rnn = nn.LSTM(64, nhid, nlayers, batch_first = True, bidirectional = False)
        self.classifier = nn.Linear(nhid,7)
        
        self.rnn_nhid = nhid
        self.rnn_layers = nlayers



    def forward(self, inputs, hidden, eval=False):
        ## Input is a sequence of faces for each user. batch dimension is in users (users, sequence, channels, height, width)
        
        #first get a slice for earch sequence element, get features from convolutional and store output in another sequence to feed the RNN
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
            rand_start = np.random.randint(rest)
        else:
            rand_start = 0

        if True:
            features = []
            for s in range(seq_window):
                #input_slice = inputs.narrow(1,s,1).contiguous().view(-1,3,224,224).cuda()
                #input_slice = inputs[:,s,:,:,:].cuda()
                input_slice = inputs[:,s+rand_start,:,:,:]
                if eval == False:
                    input_slice = torch.autograd.Variable(input_slice).cuda()
                else:
                    input_slice = torch.autograd.Variable(input_slice, volatile=True).cuda()
                slice_features = self.vgg_face(input_slice)
                features.append(slice_features)
                
            
            ## concatenate in (batch, sequence, features)
            features = torch.stack(features, dim=1)

        else:
            all_faces = inputs.narrow(1,rand_start,seq_window).contiguous().view(input_sizes[0] * seq_window, input_sizes[2], input_sizes[3], input_sizes[4])

            if eval == False:
                all_faces = torch.autograd.Variable(all_faces).cuda()
            else:
                all_faces = torch.autograd.Variable(all_faces, volatile=True).cuda()

            all_features = self.vgg_face(all_faces)
            features = all_features.view(input_sizes[0], seq_window, -1)

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
        output = torch.mean(torch.stack(outputs, dim=2), dim=2).view(-1,7)
        
        
        
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
