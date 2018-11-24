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
                temp_sentence.append([sentence[1], sentence[3]])
        else:
            sentence_list.append(temp_sentence)
            temp_sentence = []
        
    return sentence_list

# convert only lower case 
# remove single quote and the suffixes after single quote from words     
def apply_preprocessing(sentence_list):
    tag_list = defaultdict()
    word_list = defaultdict()
    pre_sentence_list = []
    for sentence in sentence_list:
        temp_list = []
        for word in sentence:        
            real_word = preprocess(word[0])
            temp_list.append([real_word, word[1]])
            tag_list[word[1]] = 1
            word_list[real_word] = 1
        pre_sentence_list.append(temp_list)
    return pre_sentence_list, tag_list, word_list

def preprocess(word):
    real_word = word.lower()
    vocabulary = re.findall(r"(\w+)'", real_word)
    if vocabulary:
        real_word = vocabulary[0]
    return real_word

def give_id(tag_list, word_list):
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
    return tags, words

def convert_id(pre_sentence_list, tags,words):
    pre_sentence = []
 
    for sentence in pre_sentence_list:
        temp = []
        for word_tag_pair in sentence:
            word_id = words[word_tag_pair[0]]
            tag_id = tags[word_tag_pair[1]]
            temp.append([word_id, tag_id])
        pre_sentence.append(temp)
    return pre_sentence

def create_matrices(pre_sentence, len_tags, len_words, apply_smoothing=True):
    # gets the number of count for tag-tag pair 
    transition_matrix = np.zeros((len_tags, len_tags))
    # gets the number of count for word-tag pair
    observation_matrix = np.zeros((len_words, len_tags))

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
    temp1 = np.zeros((1, len_tags))
    temp2 = np.zeros((len_tags+1, 1))
    for sentence in pre_sentence:
        first_tag_id = sentence[0][1]
        last_tag_id = sentence[-1][1]
        temp1[0][first_tag_id] = temp1[0][first_tag_id] + 1 
        temp2[last_tag_id][0] = temp2[last_tag_id][0] + 1 
        
    # start node in row , end node in column    
    transition_matrix = np.vstack((transition_matrix, temp1))
    transition_matrix = np.hstack((transition_matrix, temp2))
    
    count_observation = observation_matrix.copy()
    count_transition = transition_matrix.copy()
    
    observation_matrix = observation_matrix / np.sum(observation_matrix, axis=0)
    
    if not apply_smoothing:
        transition_matrix = transition_matrix / np.sum(transition_matrix, axis=1).reshape((len_tags+1, 1))
    else:  
        smoothed_count_transition = [[(element + 1) for element in vector] for vector in count_transition]
        smoothed_transition_matrix = smoothed_count_transition / (np.sum(smoothed_count_transition, axis=1).reshape((len_tags+1, 1)) + len_tags*len_tags)
        return smoothed_transition_matrix, observation_matrix, count_observation, smoothed_count_transition
    
    return transition_matrix, observation_matrix, count_observation, count_transition

def get_unknown_prob_verb(observation_matrix, len_tags, unknown_prob, tags, words, next_word_id):
    verb_index = tags['Verb']
    point_id1 = words["."]
    point_id2 = words["!"]
    point_id3 = words["?"]
    point_id4 = words[".."]
    point_id5 = words["..."]
       
    observation_value = 0
    if next_word_id == point_id1 or next_word_id == point_id2 or next_word_id == point_id3 or next_word_id == point_id4 or next_word_id == point_id5:
        observation_value = unknown_prob[verb_index]
    return observation_value




def generate_unknown_prob(observation_matrix, len_tags):
    # the number of token for each tag
    tag_sum  = np.sum(observation_matrix, axis=0)        
    temp = np.sum(observation_matrix, axis=1)
    singletons = [i for i in range(len(temp)) if temp[i]==1] # id of  singleton tokens in the training set
    num_singletons = len(singletons)
    singletons_tags = np.zeros((len_tags)) # the number of singleton for each tag
    
    for word in singletons:
        index  = list(observation_matrix[word]).index(1)
        singletons_tags[index] = singletons_tags[index] + 1

    prob_unknown = (1/num_singletons)*(singletons_tags/tag_sum)

    return prob_unknown        
 
# viterbi is used for only test sequence
# sequence consists of id of the word in the sentence    
def viterbi_algorithm(sequence, len_tags, transition_matrix, observation_matrix, tags, words, unknown_prob=None, morp_analysis = True):
    T  = len(sequence) # number of word in sequence
    word_id = sequence[0] # word_id  = id of the first word
    
    viterbi = np.zeros((len_tags+2, T+1))
    backpointer = np.zeros((len_tags+2, T+1), dtype=np.int)
    verb_index = tags['Verb']
    # start state is full
    # -1 means the start state(node) in backpointer
    for state in range(len_tags):
        observation_value = observation_matrix[word_id][state] 
        if observation_value == 0 and (not(unknown_prob is None)):
            observation_value = unknown_prob[state]
        viterbi[state][0] = transition_matrix[len_tags][state] * observation_value
        backpointer[state][0] = -1

    for t in range(1, T):
        word_id = sequence[t]
        for s in range(len_tags):
            observation_value = observation_matrix[word_id][s] 
            if observation_value == 0 and (not(unknown_prob is None)) :
                if morp_analysis  and t+1 == T-1 and s == verb_index:
                    next_word_id = sequence[t+1] 
                    observation_value = get_unknown_prob_verb(observation_matrix,len_tags,unknown_prob,tags,words,next_word_id)
                    
                else:    
                    observation_value = unknown_prob[s]
            
            temp = [viterbi[s_][t-1]*transition_matrix[s_][s]*observation_value for s_ in range(len_tags)]
     
            temp2 = [viterbi[s_][t-1]*transition_matrix[s_][s] for s_ in range(len_tags)]

            index, value = max(enumerate(temp2), key=operator.itemgetter(1))
            viterbi[s][t] = max(temp)
            backpointer[s][t] = index
        
    temp = [viterbi[s_][T-1]*transition_matrix[s_][len_tags] for s_ in range(len_tags)]
    index, value = max(enumerate(temp), key=operator.itemgetter(1))        

    # len_tag+1 means the end node of the graph
    viterbi[len_tags+1][T] = value
    backpointer[len_tags+1][T] = index

    return viterbi, backpointer

def get_POS_tags (backpointer, sequence, len_tags):
    T = len(sequence)
    last_node_id = backpointer[len_tags+1][T]
    POS_tags = [last_node_id]
    
    for i in range(T):
        previous_node_id = backpointer[last_node_id][T-1-i] 
        POS_tags.append(previous_node_id)
        last_node_id = previous_node_id
    
    return POS_tags[::-1]

def read_file():
    sentence_list = get_original_sentences(input_file)
    pre_sentence_list, tag_list, word_list = apply_preprocessing(sentence_list)
    tags, words = give_id(tag_list, word_list)
    pre_sentence = convert_id(pre_sentence_list, tags, words)
    len_tags = len(tags)
    len_words = len(words)
    pre_sentence = np.array(pre_sentence)
    return words, tags, len_tags, len_words, pre_sentence
