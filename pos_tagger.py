import numpy as np
import pandas as pd
import re
from collections import defaultdict
import operator
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


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
def preprocessing(sentence_list):
    tag_list = defaultdict()
    word_list = defaultdict()
    pre_sentence_list = []
 
    for sentence in sentence_list:
        temp_list = []
        for word in sentence:
            real_word = word[0].lower()
            vocabulary = re.findall(r"(\w+)'", real_word)
            if vocabulary:
                real_word = vocabulary[0]
                
            temp_list.append([real_word, word[1]])
            tag_list[word[1]] = 1
            word_list[real_word] = 1
        pre_sentence_list.append(temp_list)
    return pre_sentence_list, tag_list, word_list

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
def viterbi_algorithm(sequence, len_tags, transition_matrix, observation_matrix, unknown_prob=None):
    T  = len(sequence) # number of word in sequence
    word_id = sequence[0] # word_id  = id of the first word
    
    viterbi = np.zeros((len_tags+2, T+1))
    backpointer = np.zeros((len_tags+2, T+1), dtype=np.int)
    
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
            if observation_value == 0 and (not(unknown_prob is None)):
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

def test_preparation(X_test):
    sentences = []
    sentences_tags = []
    for sentence in X_test:
        words = []
        tags = []
        for pair in sentence:
            words.append(pair[0])
            tags.append(pair[1])
            
        sentences.append(words)
        sentences_tags.append(tags)
    return sentences, sentences_tags
            

def get_viterbi_test(X_test, len_tags, transition_matrix, observation_matrix, prob_unknown):
    
    X_sentences, X_tags = test_preparation(X_test)
    sentence_compare = 0
    word_compare = 0 
    result_tags = []
    count = 0 
    total_word = 0
    for sequence in X_sentences:
        viterbi, backpointer =  viterbi_algorithm(sequence, len_tags, transition_matrix, observation_matrix, prob_unknown)
        POS_tags = get_POS_tags(backpointer, sequence, len_tags)
        final_tag = POS_tags[1:]
        result_tags.append(final_tag)
        
        if X_tags[count] == final_tag:
            sentence_compare = sentence_compare + 1
            
        for i in range(len(X_tags[count])):
            total_word = total_word + 1
            if X_tags[count][i] == final_tag[i]:
                word_compare = word_compare + 1
       
        count = count + 1    
    
    sentence_result = sentence_compare/count
    word_result = word_compare/total_word
    # print("Sentence comparison: ", sentence_compare, "->", count, "%", sentence_result)
    # print("Word comparison: ", word_compare, "->", total_word, "%", word_result)
    return sentence_result, word_result



def get_evaluation(pre_sentence, len_tags, len_words):
    total_sentence = 0
    total_word = 0
    count = 10
    for i in range(count):
        # X_train, X_test = train_test_split(pre_sentence, test_size=0.1, shuffle = True)
        total_sentence_fold = 0
        total_word_fold = 0
        split_count = 6
        kf = KFold(n_splits = split_count, shuffle=True)
        kf.get_n_splits(pre_sentence)
        for train_index, test_index in kf.split(pre_sentence):
            X_train, X_test = pre_sentence[train_index], pre_sentence[test_index]
            transition_matrix, observation_matrix, count_observation, count_transition = create_matrices(X_train, len_tags, len_words)
            prob_unknown =  generate_unknown_prob(count_observation,len_tags)
            sentence_result, word_result = get_viterbi_test(X_test, len_tags, transition_matrix, observation_matrix, prob_unknown)
            total_sentence_fold += sentence_result
            total_word_fold += word_result
        avg_sentence_fold = (total_sentence_fold / split_count)
        avg_word_fold = (total_word_fold / split_count)
        total_sentence += avg_sentence_fold
        total_word += avg_word_fold
        print("Batch " + str(i))
        print("\tSentence result: ", 100 * avg_sentence_fold)
        print("\tWord result: ", 100 * avg_word_fold)
    average_sentence = (total_sentence / count)
    average_word = (total_word / count)
    print("Sentence result: ", 100 * average_sentence)
    print("Word result: ", 100 * average_word)
    
    
#%%    

sentence_list = get_original_sentences(input_file)
pre_sentence_list, tag_list, word_list = preprocessing(sentence_list)
tags, words = give_id(tag_list, word_list)
pre_sentence = convert_id(pre_sentence_list, tags, words)
len_tags = len(tags)
len_words = len(words)
pre_sentence = np.array(pre_sentence)
get_evaluation(pre_sentence, len_tags, len_words)




