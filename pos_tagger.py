import numpy as np
import pandas as pd
import re
from collections import defaultdict
import operator


input_file = 'Project (Application 1) (MetuSabanci Treebank).conll'

def get_original_sentences(input_file):
    file = open(input_file, 'r')
    sentence_list = []
    temp_sentence = []
    for line in file:
        if  line.strip() :
            sentence = re.split(r'\t+', line)
            if "_" not in sentence[1]: 
                temp_sentence.append([sentence[1],sentence[3]])
        else:
            sentence_list.append(temp_sentence)
            temp_sentence = []
        
    return sentence_list


# convert only lower case ? 
def preprocessing(sentence_list):
    tag_list = defaultdict()
    word_list = defaultdict()
    pre_sentence_list = []
 
    for sentence in sentence_list:
        temp_list = []
        for word in sentence:
            temp_list.append([word[0].lower(),word[1]])
            tag_list[word[1]] = 1
            word_list[word[0].lower()] = 1
        pre_sentence_list.append(temp_list)
    return pre_sentence_list,tag_list,word_list

def give_id(tag_list,word_list):
    tags = defaultdict()
    words = defaultdict()
    count = 0
    for tag in tag_list:
        tags[tag] = count
        count = count + 1
    count = 0
    for word in word_list:
        words[word] = count
        count = count + 1
    return tags,words

def convert_id(pre_sentence_list,tags,words):
    pre_sentence = []
 
    for sentence in pre_sentence_list:
        temp = []
        for word_tag_pair in sentence:
            word_id = words[word_tag_pair[0]]
            tag_id = tags[word_tag_pair[1]]
            temp.append([word_id,tag_id])
        pre_sentence.append(temp)
    return pre_sentence

    

def create_matrices(pre_sentence,len_tags,len_words):
    # gets the number of count for tag-tag pair 
    transition_matrix = np.zeros((len_tags,len_tags))
    # gets the number of count for word-tag pair
    observation_matrix = np.zeros((len_words,len_tags))

    for sentence in pre_sentence:
        for word_tag_pair in sentence:
            word_id = word_tag_pair[0]
            tag_id = word_tag_pair[1]
            observation_matrix[word_id][tag_id] = observation_matrix[word_id][tag_id] + 1
   
  
    for sentence in pre_sentence:
        previous_id = sentence[0][1]
        for word_tag_pair in sentence[1:]:
            current_id = word_tag_pair[1]
            transition_matrix[previous_id][current_id] = transition_matrix[previous_id][current_id] + 1
            previous_id = current_id
            
    return transition_matrix,observation_matrix


def generate_unknown_prob():
    # her tagde kac word var
    tag_sum  = np.sum(observation_matrix, axis=0)        
    temp = np.sum(observation_matrix, axis=1)
    singletons = [i for i in range(len(temp)) if temp[i]==1] # train datadaki singleton wordlerin idleri
    num_singletons = len(singletons)
    singletons_tags = np.zeros((len_tags)) # her tagdeki singleton sayısı
    
    for word in singletons:
        index  = list(observation_matrix[word]).index(1)
        singletons_tags[index] = singletons_tags[index] + 1
                
    prob_unknown = (1/num_singletons)*(singletons_tags/tag_sum)

    return prob_unknown        


    
# Not ready now
# viterbi sadece testte kullanılacak    
def viterbi_no_smoothing():
    viterbi_base_matrix = np.zeros((len_tags+2,len_tags))
    for sentence in pre_sentence:
        first_tag_id = sentence[0][1]
        last_tag_id = sentence[-1][1]
        viterbi_base_matrix[0][first_tag_id] = viterbi_base_matrix[0][first_tag_id] + 1 
        viterbi_base_matrix[1][last_tag_id] = viterbi_base_matrix[1][last_tag_id] + 1 
    
    viterbi_base_matrix[0] = viterbi_base_matrix[0] / np.sum(viterbi_base_matrix[0])
    viterbi_base_matrix[-1] = viterbi_base_matrix[-1] / np.sum(viterbi_base_matrix[-1])  






sentence_list = get_original_sentences(input_file)
pre_sentence_list,tag_list,word_list = preprocessing(sentence_list)
tags, words = give_id(tag_list,word_list)
pre_sentence = convert_id(pre_sentence_list,tags,words)
len_tags = len(tags)
len_words = len(words)

transition_matrix,observation_matrix = create_matrices(pre_sentence,len_tags,len_words)



#%%


    
    
    
    
    
    
    














