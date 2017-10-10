import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from models.multimodal import video_seq_model as v_seq_model



class Word_Embeddings_sequence_model(nn.Module):

    def __init__(self, vocab_size, embedding_size, nhid, nlayers, dropout=0.5, pretrained_embedding_path = None):
        super(Word_Embeddings_sequence_model, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.video_model = v_seq_model.Vgg_face_sequence_model(nhid=32, nlayers=2)
        
        self.rnn_nhid = nhid
        self.rnn_layers = nlayers
        self.bidirectional = False
        self.seq_window = 100

        self.rnn = nn.LSTM(embedding_size+self.video_model.face_embedding_size, nhid, nlayers, dropout=dropout, batch_first = True, bidirectional = self.bidirectional)
        #self.rnn = nn.RNN(64, nhid, nlayers, batch_first = True, bidirectional = False)
        #self.rnn = nn.LSTM(64, nhid, nlayers, batch_first = True, bidirectional = False)

        #BS
        ##A different classifier for each personality trait, since they are not mutually exclusive
        #self.classifiers = []
        #for i in range(6):
            #self.classifiers.append( nn.Linear(nhid,1))
    
        self.classifier = nn.Linear(nhid,6)



    def forward(self, transcripts, faces, hidden, eval=False):
        
        seq_length = transcripts.size()[1]
        
        face_embeddings = []
        for images in faces:
            #images=images[0:4,:,:,:,:]
            image_var = images
            #print(images.size()[0])
            face_model_hidden = self.video_model.init_hidden(images.size()[0])
            if True: #(USE_CUDA)
                face_model_hidden = (face_model_hidden[0].cuda(), face_model_hidden[1].cuda())
            face_embedding = self.video_model(image_var, face_model_hidden)
            #(pad_l, pad_r, pad_t, pad_b )
            print(face_embedding.size())
            padding_size = seq_length - face_embedding.size()[0]
            if padding_size > 0:
                zeros = torch.FloatTensor(padding_size, face_embedding.size()[1] ).zero_()
                zeros = torch.autograd.Variable(zeros).cuda() 
                print("padding size", zeros.size())
                face_embedding = torch.cat( [face_embedding, zeros], dim=0 )
            print(face_embedding.size())
            face_embeddings.append(face_embedding)
        
        face_embeddings = torch.stack(face_embeddings, dim=0)
        print(face_embeddings.size())
        
        
        #first get a slice for earch sequence element, get features from convolutional and store output in another sequence to feed the RNN
        transcripts_sizes = transcripts.size() 
        seq_length = transcripts_sizes[1]
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

        cropped_input = transcripts.narrow(1,rand_start,seq_window).contiguous().view(transcripts_sizes[0], seq_window)

        if eval == False:
            cropped_input = torch.autograd.Variable(cropped_input).cuda()
        else:
            cropped_input = torch.autograd.Variable(cropped_input, volatile=True).cuda()

        embeddings = self.embedding(cropped_input)
        print("embeddings size", embeddings.size())
        embeddings = torch.cat([embeddings, face_embeddings], dim=2) 

        ## feed to the RNN
        output, hidden = self.rnn(embeddings, hidden)
        ## since we set batch_first=True, output size is (batch, seq_length, hidden_size * num_directions)
                
        n_outputs = output.size()[1]
        outputs = []
        for o in range(n_outputs):
            outputs.append( self.classifier(output[:,o,:])  )
        final_output = torch.mean(torch.stack(outputs, dim=2), dim=2).view(-1,6)

        
        
        #total_output = []
        #for classifier in self.classifiers:
            #outputs = []
            #for o in range(n_outputs):
                #outputs.append( classifier(output[:,o,:])  )
            #output = torch.mean(torch.stack(outputs, dim=2), dim=2).view(-1,1)
            #total_output.append(output)
        #final_output = torch.stack(total_output, dim=2).view(-1,6)
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


