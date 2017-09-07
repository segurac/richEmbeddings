import pickle
import sys


#transcriptions_path = 'transcription_training.pkl'
transcriptions_path = sys.argv[1]

with open(transcriptions_path, 'rb') as stream:
  transcriptions = pickle.load(stream)#, encoding='latin1')
    
for video in transcriptions.keys():
  with open('transcripts/' + video + '.txt', 'w') as stream:
    stream.write(transcriptions[video].strip())
    
