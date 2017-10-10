import io
import torch
from torchvision import models, transforms, datasets
import torch.utils.data as data
import numpy as np

import os
import os.path
import pickle
from PIL import Image
import hashlib

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
  

class MultimodalReader(data.Dataset):
  
    def __init__(self, annotations_path, transcriptions_path, fbank_path, faces_path, transform=None, target_transform=None):
        
        
        self.fbank_path = fbank_path
        self.faces_path = faces_path
        self.transform = transform
        self.target_transform = target_transform
        
        with open(annotations_path, 'rb') as stream:
            self.annotations = pickle.load(stream, encoding='latin1')

        self.videos = list(self.annotations['openness'].keys())
        self.traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism", "interview"] #self.annotations.keys()
        
        self.word2id = {}
        self.id2word = {} 
        self.transcriptions_id = {}  
        self.transcriptions_ts = {}
        self.video_sequences ={}
        
        self.padding = self.getWordId('PADDING_TOKEN')
        self.unknownToken = self.getWordId('UNKNOWN_TOKEN')
        
        print("Reading transcriptions")
        self.read_all_ctms(transcriptions_path)
        print("Scanning faces")
        self.build_faces_dir_tree(faces_path)
        print("Reading video metada")
        self.read_video_metadata(transcriptions_path + '/../../')
        print("Segmenting audio and video according to ctms")
        self.assing_frames_to_words(faces_path, fbank_path)
        
       
       
    def build_faces_dir_tree(self, faces_path):
        self.video_sequences ={}
        for video_id in self.videos:
            images = []
            video_path = os.path.join(faces_path, video_id)
            for root, _, fnames in sorted(os.walk(video_path)):
              for fname in sorted(fnames):
                  if is_image_file(fname):
                      path = os.path.join(root, fname)
                      images.append(path)
            self.video_sequences[video_id] = images
        
    def load_ctm(self, ctm_path):
        data = []
        try:
            with open(ctm_path, 'r') as stream:
                for line in stream:
                    ctm_line = line.strip().split()
                    spk_id = ctm_line[0]
                    nothing = ctm_line[1]
                    start_ts = ctm_line[2]
                    end_ts = ctm_line[3]
                    word = ctm_line[4]
                    if len(ctm_line) == 6:
                        confidence = ctm_line[5]
                    else:
                        confidence = 1.0
                    new_data = {}
                    new_data['start_ts'] = float(start_ts)
                    new_data['end_ts'] = float(end_ts) + float(start_ts)
                    if word == "<unk>":
                        word = "UNKNOWN_TOKEN"
                    new_data['word'] = str(word)
                    data.append(new_data)
        except:
            ## test reading the ASR transciption
            try:
                dir_path =  os.path.dirname(ctm_path)
                ctm_name = os.path.basename(ctm_path)
                md5sum = ("../" + ctm_name.replace(".ctm.clean","") + ".wav\n")
                #print(md5sum)
                md5sum = hashlib.md5(md5sum.encode()).hexdigest()
                new_ctm_path = dir_path + "/../ctm_asr/" + md5sum + ".ctm"
                print("CTM file not found", ctm_path)
                print("Trying with", new_ctm_path)
                
                with open(new_ctm_path, 'r') as stream:
                    for line in stream:
                        #print(line)
                        ctm_line = line.strip().split()
                        spk_id = ctm_line[0]
                        start_ts = ctm_line[1]
                        end_ts = ctm_line[2]
                        word = ctm_line[3]
                        confidence = ctm_line[4]

                        new_data = {}
                        new_data['start_ts'] = float(start_ts)
                        new_data['end_ts'] = float(end_ts) + float(start_ts)
                        if word == "<unk>":
                            word = "UNKNOWN_TOKEN"
                        new_data['word'] = str(word)
                        data.append(new_data)
            except:
                #print("CTM file not found", ctm_path)
                #print("CTM file from ASR not found", new_ctm_path)
                print("Creating one single <unk> spanning the first 10 seconds")
                new_data = {}
                new_data['start_ts'] = float(0.0)
                new_data['end_ts'] = float(10.0) + float(0.0) 
                new_data['word'] = "UNKNOWN_TOKEN"
                data.append(new_data)
        return data
        
        
    def read_all_ctms(self, transcriptions_path):

        self.transcriptions_id = {}  
        self.transcriptions_ts = {}
        for video_id in self.videos:
            word_ids = []
            word_ts = []
            ctm_data = self.load_ctm( transcriptions_path + '/' + video_id + '.ctm.clean')
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
    
    def assing_frames_to_words(self, faces_path, audio_path,  audio_frame_step=100.0, desired_video_frame_rate = 25.0):
        self.video_frames = {}
        self.audio_frames = {}
        for video_id in self.videos:
            all_word_faces = []
            all_word_audio_frames = []
            #get the set of frames from the list of images
            frame_set = set( self.video_sequences[video_id] )
            for i in range( len( self.transcriptions_id[video_id] )):
                word_faces = []
                start_ts = self.transcriptions_ts[video_id][i]['start_ts']
                end_ts   = self.transcriptions_ts[video_id][i]['end_ts']
                real_fps = self.video_fps[video_id]
                
                video_frame_start = np.floor( start_ts * desired_video_frame_rate)
                video_frame_end = np.ceil (end_ts * desired_video_frame_rate) + 1
                for video_frame in range(int(video_frame_start), int(video_frame_end)):
                    resampled_video_frame = int(round(video_frame * real_fps / desired_video_frame_rate))
                    #generar el path completo al frame
                    frame_path = os.path.join(faces_path, video_id, 'I_' + str(1000 + resampled_video_frame) + '.jpg')
                    #print(frame_path)
                    #mirar si el path está en la sequencia de frames y ponerlo en la lista
                    if frame_path in frame_set:
                        word_faces.append(frame_path)
                    #si no está poner un string vacío
                    else:
                        word_faces.append('')
                all_word_faces.append(word_faces)
                
                audio_frame_start = int(np.floor( start_ts * audio_frame_step))
                audio_frame_end = int(np.ceil (end_ts * audio_frame_step))
                timestamps = {}
                timestamps['start_frame'] = audio_frame_start
                timestamps['end_frame'] = audio_frame_end
                all_word_audio_frames.append(timestamps)
                
                
            self.video_frames[video_id] = all_word_faces
            self.audio_frames[video_id] = all_word_audio_frames
            
            
                    
        
    
    def read_video_metadata(self, metadata_path):
        self.video_duration ={}
        self.video_fps = {}
        
        with open(metadata_path + '/video_duration.txt','r') as stream:
            for line in stream:
                video_id, duration = line.strip().split()
                duration = float(duration)
                self.video_duration[video_id] = duration
                
        with open(metadata_path + '/video_fps.txt','r') as stream:
            for line in stream:
                video_id, fps = line.strip().split()
                fps = float(fps)
                self.video_fps[video_id] = fps
        
    

    def __len__(self):
        return len(self.videos)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: ([word_ids], [target]) where target is the list of scores of each of the personality traits.
        """

        video_id = self.videos[index]
        transcript = self.transcriptions_id[video_id]
        
        
        labels = []
        for trait in self.traits:
            score = self.annotations[trait][video_id]
            labels.append(score)
            
            
        #load audio, un numpy array de dimensión variable para cada palabra dependiendo de los frames de comienzo y final
        audio_filterbank_file = self.fbank_path + "/" + video_id + ".wav.fbank.pickle"
        with open(audio_filterbank_file, 'rb') as stream:
            all_fb_data = pickle.load(stream)
        
        audio_data = []
        for i, timestamps in enumerate(self.audio_frames[video_id]):
            start_frame = timestamps['start_frame']
            end_frame = timestamps['end_frame']
            audio_data.append(all_fb_data[start_frame:end_frame])
            
        #load video, una lista de imágenes para cada palabra
        #video_data = []
        #for sequence in self.video_frames[video_id]:
            #sequence_data = []
            #for img_path in sequence:
                #img = default_loader(img_path)
                #if self.transform is not None:
                    #img = self.transform(img)
                #sequence_data.append(img)
            #video_data.append(sequence_data)

        video_data = []
        for sequence in self.video_frames[video_id]:
            sequence_data = []
            for img_path in sequence:
                with open(img_path, 'rb') as stream:
                    img = pickle.load(stream)                
                sequence_data.append(img)
            video_data.append(sequence_data)
        
        return([transcript, audio_data, video_data], labels)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader2(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

    
def default_loader(path):
    return Image.open(path).convert('RGB')



def my_collate(batch):
    
    
    max_lengths_video = []
    max_lengths_audio = []
    for n in range(len(batch)):
        transcripts = batch[n][0][0]
        audio       = batch[n][0][1]
        video       = batch[n][0][2]
        target      = batch[n][1] 
    
        max_length=0
        for i in range(len(video)):
            nphotos = len(video[i])
            if nphotos > max_length:
                max_length = nphotos
        max_lengths_video.append(max_length)
        
        max_length = 0
        for i in range(len(audio)):
            nframes = audio[i].shape[0]
            if nframes > max_length:
                max_length = nframes
        max_lengths_audio.append(max_length)
            
    max_lengths_audio = np.array(max_lengths_audio)
    max_lengths_video = np.array(max_lengths_video)
    max_length_audio = np.max(max_lengths_audio)
    max_length_video = np.max(max_lengths_video)
    
    max_length_transcripts=0
    for n in range(len(batch)):
        nwords = len(batch[n][0][0])
        if nwords > max_length_transcripts:
            max_length_transcripts = nwords
    

    ### Prepare transcripts data tensor, one big matrix (batch_size, max_length_transcripts)
    transcripts_data_tensor = torch.LongTensor(
            len(batch), 
            max_length_transcripts
            ).zero_()
    print(transcripts_data_tensor.size())

    for n in range(len(batch)):
        nwords = len(batch[n][0][0])
        for word_idx in range(nwords):
            transcripts_data_tensor[n][word_idx] = batch[n][0][0][word_idx]

    ### Prepare target variables, one big matrix, (batch_size, 5 traits + 1 interview)
    target = torch.FloatTensor( len(batch), 6).zero_()
    for n in range(len(batch)):
        for trait in range(len(batch[n][1])):
            target[n][trait] = batch[n][1][trait]

    ### Prepare video data tensor, array of tensors, (nwords, sample_max_seq_length, f)
    total_video_data = []
    for n in range(len(batch)):
        video = batch[n][0][2]
        nwords = len(video)
        #nchannels = video[0][0].size()[0]  #[channels, width?, height?]
        feat_size = video[0][0].shape[0]
        
        video_data_tensor = torch.FloatTensor(nwords, int(max_lengths_video[n]), feat_size ).zero_()
        ### fill in data
        for word in range(nwords):
            for i, img in enumerate(video[word]):
                video_data_tensor[word,i] = torch.from_numpy(img)
        total_video_data.append(video_data_tensor)
        
        
    ### Prepare audio data tensor, array of tensors, (nwords, sample_max_seq_length, n_filterbanks)
    total_audio_data = []
    for n in range(len(batch)):
        audio = batch[n][0][1]
        nwords = len(audio)
        n_filterbanks = audio[0].shape[1]
        print(n_filterbanks)
        audio_data_tensor = torch.FloatTensor(nwords, int(max_lengths_audio[n]), n_filterbanks)
        for word in range(nwords):
            for i, frame in enumerate(audio[word]):
                audio_data_tensor[word,i] = torch.from_numpy(frame)
        total_audio_data.append(audio_data_tensor)
    
    return ((transcripts_data_tensor, total_video_data, total_audio_data, target))
