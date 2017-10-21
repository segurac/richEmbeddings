import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from models.multimodal import video_seq_model as v_seq_model
from models.multimodal import audio_seq_model as a_seq_model




class Word_Embeddings_sequence_model(nn.Module):

    def __init__(self, vocab_size, embedding_size, nhid, nlayers, dropout=0.5, pretrained_embedding_path = None):
        super(Word_Embeddings_sequence_model, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.video_model = v_seq_model.Vgg_face_sequence_model(nhid=32, nlayers=1)
        self.audio_model = a_seq_model.AudioFB_sequence_model(nhid=32, nlayers=1)
        
        self.rnn_nhid = nhid
        self.rnn_layers = nlayers
        self.bidirectional = False
        self.seq_window = 100

        self.rnn = nn.LSTM(
            embedding_size
            +self.video_model.face_embedding_size 
            + self.audio_model.audio_embedding_size, 
            nhid, 
            nlayers, 
            dropout=dropout, 
            batch_first = True, 
            bidirectional = self.bidirectional
            )
        #self.rnn = nn.RNN(64, nhid, nlayers, batch_first = True, bidirectional = False)
        #self.rnn = nn.LSTM(64, nhid, nlayers, batch_first = True, bidirectional = False)

        #BS
        ##A different classifier for each personality trait, since they are not mutually exclusive
        #self.classifiers = []
        #for i in range(6):
            #self.classifiers.append( nn.Linear(nhid,1))
    
        self.classifier = nn.Linear(nhid,6)
        
        self.drop_p = 0.5
        self.fdrop = (1.0-self.drop_p)
        self.embedding_dropout = torch.nn.Dropout2d(p=self.drop_p)



    def forward(self, transcripts, faces, filterbanks, hidden, train_text=True, train_audio=True, train_video=True, eval=False):
        
        seq_length = transcripts.size()[1]
        
        
        if self.training == True:
            if train_audio:
                self.audio_model.train()
            else:
                self.audio_model.eval()
            if train_video:
                self.video_model.train()
            else:
                self.video_model.eval()
            if train_text:
                self.embedding.train()
            else:
                self.embedding.eval()
        else:
            self.audio_model.eval()
            self.video_model.eval()
            self.embedding.eval()
            
        
        
        
        audio_embeddings = []
        for person in filterbanks:
            #person shape is nwords, framelength, 26 filters)
            #images=images[0:4,:,:,:,:]
            images = person.unsqueeze(1)
            #print(images.size()[0])
            try:
                audio_model_hidden = self.audio_model.init_hidden(images.size()[0])
            except:
                print(images)
                import sys
                sys.exit()
            if True: #(USE_CUDA)
                audio_model_hidden = (audio_model_hidden[0].cuda(), audio_model_hidden[1].cuda())
            audio_embedding = self.audio_model(images, audio_model_hidden)
            #(pad_l, pad_r, pad_t, pad_b )
            #print(audio_embedding.size(), "audio_embedding")
            padding_size = seq_length - audio_embedding.size()[0]
            if padding_size > 0:
                zeros = torch.FloatTensor(padding_size, audio_embedding.size()[1] ).zero_()
                zeros = torch.autograd.Variable(zeros).cuda() 
                #print("padding size", zeros.size())
                audio_embedding = torch.cat( [audio_embedding, zeros], dim=0 )
            #print(audio_embedding.size())
            audio_embeddings.append(audio_embedding)
        
        audio_embeddings = torch.stack(audio_embeddings, dim=0)
        
        
        
        face_embeddings = []
        for images in faces:
            #images=images[0:4,:,:,:,:]
            image_var = images
            #print(images.size()[0])
            try:
                face_model_hidden = self.video_model.init_hidden(images.size()[0])
            except:
                print(images)
                import sys
                sys.exit()
            if True: #(USE_CUDA)
                face_model_hidden = (face_model_hidden[0].cuda(), face_model_hidden[1].cuda())
            face_embedding = self.video_model(image_var, face_model_hidden)
            #(pad_l, pad_r, pad_t, pad_b )
            #print(face_embedding.size())
            padding_size = seq_length - face_embedding.size()[0]
            if padding_size > 0:
                zeros = torch.FloatTensor(padding_size, face_embedding.size()[1] ).zero_()
                zeros = torch.autograd.Variable(zeros).cuda() 
                #print("padding size", zeros.size())
                face_embedding = torch.cat( [face_embedding, zeros], dim=0 )
            #print(face_embedding.size())
            face_embeddings.append(face_embedding)
        
        face_embeddings = torch.stack(face_embeddings, dim=0)
        #print(face_embeddings.size())
        
        
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
        
        #word dropout
        if self.training:
        #if False:
            unknownToken = 1
            set_word_to_unknown = np.random.uniform(size=cropped_input.size()) > 0.95
            for i in range(cropped_input.size()[0]):
                for j in range(cropped_input.size()[1]):
                    if set_word_to_unknown[i,j]:
                        cropped_input[i,j] = unknownToken

        if eval == False:
            cropped_input = torch.autograd.Variable(cropped_input).cuda()
        else:
            cropped_input = torch.autograd.Variable(cropped_input, volatile=True).cuda()

        embeddings = self.embedding(cropped_input)
        #print("embeddings size", embeddings.size())
        #dropout entire words for face_embedding and audio_embedding

        #if self.training:
        #if False:
            #s = face_embeddings.size()
            #face_embeddings= self.embedding_dropout(face_embeddings.unsqueeze(2)).view(s) * self.fdrop
            
            #s = audio_embeddings.size()
            #audio_embeddings= self.embedding_dropout(audio_embeddings.unsqueeze(2)).view(s) * self.fdrop
        
        #if False:
        ##if self.training:
            ##other possibility is to put embedding to noise
            #for the_embedding in [face_embeddings, audio_embeddings]:
                #s = the_embedding.size()

                #mean_embedding = the_embedding.view( (s[0]*s[1], s[2]) ).mean(0)
                #std_embedding  = the_embedding.view((s[0]*s[1], s[2])).std(0)
                #set_word_to_unknown = np.random.uniform(size=cropped_input.size()) > self.drop_p
                #for ii in range( s[0] ):
                    #for jj in range( s[1] ):
                        #if set_word_to_unknown[i,j]:
                            #kk = (torch.normal( mean_embedding, std_embedding * 0.3)[0]).data.cpu().numpy()
                            #kk2 = torch.FloatTensor(kk).cuda()
                            #the_embedding[ii,jj] = kk2 
        
        embeddings = torch.cat([embeddings, face_embeddings, audio_embeddings], dim=2)
        #embeddings = torch.cat([face_embeddings], dim=2) 
        #np_emb = embeddings.cpu()
        #with open('embedding.pickle','wb') as stream:
            #import pickle
            #pickle.dump(np_emb, stream)
        #with open('petazeta','rb') as stream:
            #pickle.load(stream)
        
        
        #print("Extended embeddings size", embeddings.size())

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


