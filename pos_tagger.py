import numpy as np
import pandas as pd
import re
from collections import defaultdict
import operator
from sklearn.model_selection import train_test_split


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


    
    #add start and end node in transition matrix
    temp1 = np.zeros((1,len_tags))
    temp2 = np.zeros((len_tags+1,1))
    for sentence in pre_sentence:
        first_tag_id = sentence[0][1]
        last_tag_id = sentence[-1][1]
        temp1[0][first_tag_id] = temp1[0][first_tag_id] + 1 
        temp2[last_tag_id][0] = temp2[last_tag_id][0] + 1 
        
    # start node in row , end node in column    
    transition_matrix = np.vstack((transition_matrix, temp1))
    transition_matrix = np.hstack((transition_matrix, temp2))
    
    # convert probability matrix
    observation_matrix = observation_matrix / np.sum(observation_matrix, axis=0)
    transition_matrix = transition_matrix / np.sum(transition_matrix, axis=1).reshape((len_tags+1,1))

            
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


    
# viterbi is used for only test sequence
# sequence consists of id of the word in the sentence    
def viterbi_algorithm(sequence):
    T  = len(sequence) # number of word in sequence
    word_id = sequence[0] # word_id  = id of the first word
    
    viterbi = np.zeros((len_tags+2,T+1))
    backpointer = np.zeros((len_tags+2,T+1))
    
    # start state is full
    # -1 means the start state(node) in backpointer
    for state in range(len_tags):
        viterbi[state][0] = transition_matrix[len_tags][state] * observation_matrix[word_id][state]
        backpointer[state][0] = -1

    for t in range(1,T):
        word_id = sequence[t]
        for s in range(len_tags):
            temp = []
            temp2 = []
            for s_ in range(len_tags):
                temp.append(viterbi[s_][t-1]*transition_matrix[s_][s]*observation_matrix[word_id][s])
                temp2.append(viterbi[s_][t-1]*transition_matrix[s_][s])
            
            index, value = max(enumerate(temp2), key=operator.itemgetter(1))
            viterbi[s][t] = max(temp)
            backpointer[s][t] = index
        
    temp2 = []
    for s_ in range(len_tags):
        temp2.append(viterbi[s_][T-1]*transition_matrix[s_][len_tags])
    
    
    index, value = max(enumerate(temp2), key=operator.itemgetter(1))        

    # len_tag+1 means the end node of the graph
    viterbi[len_tags+1][T] = value
    backpointer[len_tags+1][T] = index
    backpointer = np.int64(backpointer)


    return viterbi , backpointer

def get_POS_tags (backpointer,sequence,len_tags):
    T = len(sequence)
    last_node_id = backpointer[len_tags+1][T]
    POS_tags = [last_node_id]
    
    for i in range(T):
        previous_node_id = backpointer[last_node_id][T-1-i] 
        POS_tags.append(previous_node_id)
        last_node_id = previous_node_id
    
    return POS_tags[::-1]

# only works for an example , but it will improves later
def get_example(input_file):
    sentence_list = get_original_sentences(input_file)
    pre_sentence_list,tag_list,word_list = preprocessing(sentence_list)
    tags, words = give_id(tag_list,word_list)
    pre_sentence = convert_id(pre_sentence_list,tags,words)
    len_tags = len(tags)
    len_words = len(words)
    transition_matrix,observation_matrix = create_matrices(pre_sentence,len_tags,len_words)
    
    # sequence example
    sequence = [2753,17002]
    viterbi , backpointer =  viterbi_algorithm(sequence)
    POS_tags = get_POS_tags (backpointer,sequence,len_tags)
    print(POS_tags)    

get_example(input_file)


#%%    










