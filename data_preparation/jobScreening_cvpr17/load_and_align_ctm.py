#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import alignment as ali


def get_list_of_words_from_ctm(ctm, spk_id):
    data= ctm[spk_id]
    out_word_list = []
    for d in data:
        out_word_list += [d['word']]

    return out_word_list

def load_global_ctm_file(filename):
    
    #print("Loading ", filename)
    
    ctm_data = {}
    old_spk_id=''
    data = []
    with open(filename, 'r') as stream:
        for line in stream:
            spk_id, nothing, start_ts, end_ts, word = line.strip().split()
            

            if spk_id != old_spk_id:
                #save data to ctm_data
                if old_spk_id != '':
                  ctm_data[old_spk_id] = data
                data = []
                old_spk_id = spk_id
            
            new_data = {}
            new_data['start_ts'] = float(start_ts)
            new_data['end_ts'] = float(end_ts) + float(start_ts) 
            new_data['word'] = str(word)
            data.append(new_data)
            
        ctm_data[spk_id] = data
            
    return ctm_data
  

def load_transcription_from_text_file(file_id, dir):
    with open( dir + '/' + file_id + '.txt.clean', 'r') as stream:
        transcript = stream.read()
        return transcript.strip().split()
      
def replace_ctm_transcription_with_real( data, aligned_real, aligned_ctm):
    data_out = []
    if len(data) != len(aligned_ctm):
        print("Error")
    else:
        for i, d in enumerate(data):
            if not d['word'].startswith('['):
                d['word'] = aligned_real[i]
                data_out.append(d)
    return data_out

def write_ctm_data_to_file(ctm_data, key, out_dir):
    filename = out_dir + '/' + key + '.ctm.clean'
    with open(filename, 'w') as stream:
        for d in ctm_data:
            s = str(key) + '\tA\t'+ str(d['start_ts']) +'\t' +  str(d['end_ts'] - d['start_ts']) + '\t' + d['word']
            print( s, file=stream)


if __name__ == "__main__":
    ctm_file_name = sys.argv[1]
    transcripts_dir = sys.argv[2]
    ctms_dir = sys.argv[3]
    global_ctm = load_global_ctm_file( ctm_file_name )
    
    for key in global_ctm.keys():
        ctm_transcript = get_list_of_words_from_ctm(global_ctm, key)
        real_transcript = load_transcription_from_text_file(key, transcripts_dir) 
        [aligned_read_transcript, aligned_ctm_transcript ] = ali.needle( real_transcript, ctm_transcript)
        cleaned_and_aligned_data = replace_ctm_transcription_with_real(global_ctm[key], aligned_read_transcript, aligned_ctm_transcript)
        #print(cleaned_and_aligned_data)
        write_ctm_data_to_file(cleaned_and_aligned_data, key, ctms_dir)
      
    
    
