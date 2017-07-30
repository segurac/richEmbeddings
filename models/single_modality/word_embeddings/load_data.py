import io
import torch
from torchvision import models, transforms, datasets
import torch.utils.data as data
import numpy as np

import os
import os.path
import pickle



class TranscriptionsReader(data.Dataset):


    def __init__(self, transcriptions_path, annotations_path):
        with open(transcriptions_path, 'rb') as stream:
            self.transcriptions = pickle.load(stream)
            
        with open(annotations_path, 'rb') as stream:
            self.annotations = pickle.load(stream, encoding='latin1')

        self.videos = list(self.transcriptions.keys())
        self.traits = self.annotations.keys()
    
    
        self.word2id = {}
        self.id2word = {} 
        
        self.padding = self.getWordId('PADDING_TOKEN')
        self.unknownToken = self.getWordId('UNKNOWN_TOKEN')
        
        self.transcriptions_id = {}
        for video in self.transcriptions.keys():
            word_ids = []
            for word in self.transcriptions[video].strip().replace(',','').replace('.','').split():
                word_id = self.getWordId(word)
                word_ids.append(word_id)
                #print(word, word_id, self.id2word[word_id])
            self.transcriptions_id[video]=word_ids
            

    def __len__(self):
        return len(self.videos)

    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: ([word_ids], [target]) where target is the list of scores of each of the personality traits.
        """
 
        video = self.videos[index]
        transcript = self.transcriptions_id[video]
        labels = []
        for trait in self.traits:
            score = self.annotations[trait][video]
            labels.append(score)
        
        return(transcript, labels)

    def getWordId(self, word, create=True):
        """Get the id of the word (and add it to the dictionary if not existing). If the word does not exist and
        create is set to False, the function will return the unknownToken value
        Args:
            word (str): word to add
            create (Bool): if True and the word does not exist already, the world will be added
        Return:
            int: the id of the word created
        """
        # Should we Keep only words with more than one occurrence ?

        word = word.lower()  # Ignore case

        # Get the id if the word already exist
        wordId = self.word2id.get(word, -1)
        
        # If not, we create a new entry
        if wordId == -1:
            if create:
                wordId = len(self.word2id)
                self.word2id[word] = wordId
                self.id2word[wordId] = word

            else:
                wordId = self.unknownToken
        
        return wordId
    
    def my_collate(batch):
        #with open('batch.pickle','wb') as stream:
            #pickle.dump(batch, stream)
            #return batch
        _prueba_batch = batch
        max_length=0
        for n in range(len(_prueba_batch)):
            nwords = len(_prueba_batch[n][0])
            if nwords > max_length:
                max_length = nwords

        data_tensor = torch.LongTensor(
            len(_prueba_batch), 
            max_length
            ).zero_()
        data_tensor.size()

        for n in range(len(_prueba_batch)):
            nwords = len(_prueba_batch[n][0])
            for word_idx in range(nwords):
                data_tensor[n][word_idx] = _prueba_batch[n][0][word_idx]

        target = torch.FloatTensor( len(_prueba_batch), 6).zero_()
        for n in range(len(_prueba_batch)):
            for trait in range(len(_prueba_batch[n][1])):
                target[n][trait] = _prueba_batch[n][1][trait]


        return((data_tensor, target))
    
