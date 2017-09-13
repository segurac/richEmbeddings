import io
import torch
from torchvision import models, transforms, datasets
import torch.utils.data as data
import numpy as np

import os
import os.path
import pickle


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
  

class MultimodalReader(data.Dataset):
  
    def __init__(self, annotations_path, transcriptions_path, fbank_path, faces_path):
        
        with open(annotations_path, 'rb') as stream:
            self.annotations = pickle.load(stream, encoding='latin1')

        self.videos = list(annotations['openness'].keys())
        self.traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism", "interview"] #self.annotations.keys()
        
        self.word2id = {}
        self.id2word = {} 
        self.transcriptions_id = {}  
        self.transcriptions_ts = {}
        self.video_sequences ={}
        
        self.padding = self.getWordId('PADDING_TOKEN')
        self.unknownToken = self.getWordId('UNKNOWN_TOKEN')
        
        self.read_all_ctms(transcriptions_path)

        
       
       
    def build_video_dir_tree(self, videos_path):
        self.video_sequences ={}
        for video_id in self.videos:
            images = []
            video_path = os.path.join(videos_path, video_id)
            for root, _, fnames in sorted(os.walk(video_path)):
              for fname in sorted(fnames):
                  if is_image_file(fname):
                      path = os.path.join(root, fname)
                      images.append(path)
            self.video_sequences[video_id] = images
        
    def load_ctm(self, ctm_path):
        data = []
        with open(filename, 'r') as stream:
            for line in stream:
                spk_id, nothing, start_ts, end_ts, word = line.strip().split()
                new_data = {}
                new_data['start_ts'] = float(start_ts)
                new_data['end_ts'] = float(end_ts) + float(start_ts) 
                new_data['word'] = str(word)
                data.append(new_data)
                
        return data
        
        
    def read_all_ctms(self, transcriptions_path):

        self.transcriptions_id = {}  
        self.transcriptions_ts = {}
        for video_id in self.videos:
            word_ids = []
            word_ts = []
            ctm_data = load_ctm( transcriptions_path + '/' + video_id + '.ctm.clean')
            for word in ctm_data:
                word_id = self.getWordId(word['word'])
                word_ids.append(word_id)
                timestamps = {}
                timestamps['start_ts'] = word['start_ts']
                timestamps['end_ts'] = word['end_ts']
                word_ts.append(timestamps)
                
                #print(word, word_id, self.id2word[word_id])
            self.transcriptions_id[video_id]=word_ids
            self.transcriptions_ts[video_id] = word_ts
        
        
        
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
    
    def assing_frames_to_words(self, audio_frame_step=100.0, desired_video_frame_rate = 25.0)
        self.video_frames = {}
        self.audio_frames = {}
        for video_id in self.videos:
            for i in range( len( self.transcriptions_id[video_id] )):
                start_ts = self.transcriptions_ts[video_id][i]['start_ts']
                end_ts   = self.transcriptions_ts[video_id][i]['end_ts']
                real_fps = self.video_fps[video_id]
                
                video_frame_start = floor( start_ts * desired_video_frame_rate)
                video_frame_end = ceil (end_ts * desired_video_frame_rate) + 1
                for video_frame in range(video_frame_start, video_frame_end):
                    resampled_video_frame = round(video_frame * real_fps / desired_video_frame_rate)
                    #generar el path completo al frame
                    #mirar si el path está en la sequencia de frames y ponerlo en la lista
                    #si no está poner un string vacío
                    
        
    

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
