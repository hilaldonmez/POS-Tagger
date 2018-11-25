import sys
import pos_tagger
import warnings
from sklearn.model_selection import KFold

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

def get_viterbi_test(X_test, len_tags, transition_matrix, observation_matrix, count_observation, tags, words ,prob_unknown, morp_analysis ):
    X_sentences, X_tags = test_preparation(X_test)
    sentence_compare = 0
    word_compare = 0 
    result_tags = []
    count = 0 
    total_word = 0
    for sequence in X_sentences:
        viterbi, backpointer = pos_tagger.viterbi_algorithm(sequence, len_tags, transition_matrix, observation_matrix, count_observation, tags, words ,prob_unknown, morp_analysis )
        POS_tags = pos_tagger.get_POS_tags(backpointer, sequence, len_tags)
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

def get_evaluation(pre_sentence, tags, words, len_tags, len_words):
    total_sentence = 0
    total_word = 0
    for i in range(BATCH_COUNT):
        # X_train, X_test = train_test_split(pre_sentence, test_size=0.1, shuffle = True)
        total_sentence_fold = 0
        total_word_fold = 0
        kf = KFold(n_splits = CROSS_VAL_BATCH_COUNT, shuffle=True)
        kf.get_n_splits(pre_sentence)
        for train_index, test_index in kf.split(pre_sentence):
            X_train, X_test = pre_sentence[train_index], pre_sentence[test_index]
            transition_matrix, observation_matrix, count_observation, count_transition = pos_tagger.create_matrices(X_train, len_tags, len_words)
            prob_unknown =  pos_tagger.generate_unknown_prob_hapax(count_observation,len_tags)
            sentence_result, word_result = get_viterbi_test(X_test, len_tags, transition_matrix, observation_matrix, count_observation, tags, words, prob_unknown, False)
            total_sentence_fold += sentence_result
            total_word_fold += word_result
        avg_sentence_fold = (total_sentence_fold / CROSS_VAL_BATCH_COUNT)
        avg_word_fold = (total_word_fold / CROSS_VAL_BATCH_COUNT)
        total_sentence += avg_sentence_fold
        total_word += avg_word_fold
        print("Batch " + str(i + 1))
        print("\tSentence result: ", 100 * avg_sentence_fold)
        print("\tWord result: ", 100 * avg_word_fold)
    average_sentence = (total_sentence / BATCH_COUNT)
    average_word = (total_word / BATCH_COUNT)
    print("Sentence result: ", 100 * average_sentence)
    print("Word result: ", 100 * average_word)

warnings.filterwarnings("ignore")

BATCH_COUNT = 10
CROSS_VAL_BATCH_COUNT = 6

arguments = sys.argv
for index, arg in enumerate(arguments):
    if arg == "-batch":
        if (index < len(arguments) - 1) and arguments[index + 1].isdigit():
            BATCH_COUNT = int(arguments[index + 1])
    elif arg == "-cross-val-batch":
        if (index < len(arguments) - 1) and arguments[index + 1].isdigit():
            CROSS_VAL_BATCH_COUNT = int(arguments[index + 1])

print("Batch count: " + str(BATCH_COUNT))
print("Batch count for cross validation: " + str(CROSS_VAL_BATCH_COUNT))

word_list, tag_list, len_tags, len_words, pre_sentence = pos_tagger.read_file()
get_evaluation(pre_sentence, tag_list , word_list, len_tags, len_words)