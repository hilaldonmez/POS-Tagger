import pos_tagger
import warnings
from collections import defaultdict

def add_to_dict(word):
	count = len(word_list.keys())
	word_list[word] = count

# does not consider ...
# does not consider numbers
def preprocess_input(input_sentence):
	punctuations = [".", ",", "\"", "?", "!", ":", "(", ")", "\'", ";"]
	sentence = input_sentence
	for punc in punctuations:
		sentence = sentence.replace(punc, (" " + punc + " "))
	tokens = sentence.split()
	tokens = [pos_tagger.preprocess(word) for word in tokens]
	return tokens

def get_viterbi_result(tokens, len_tags, transition_matrix, observation_matrix, prob_unknown):
	sequence = []
	result = ""
	for token in tokens:
		if token not in word_list.keys():
			words = add_to_dict(token)
		sequence.append(word_list[token])
	viterbi, backpointer = pos_tagger.viterbi_algorithm(sequence, len_tags, transition_matrix, observation_matrix, prob_unknown)
	POS_tags = pos_tagger.get_POS_tags(backpointer, sequence, len_tags)
	final_tag = POS_tags[1:]
	for tag in final_tag:
		result += tag_list[tag] + " "
	print(result)

def get_tag(pre_sentence, len_tags, len_words):
	transition_matrix, observation_matrix, count_observation, count_transition = pos_tagger.create_matrices(pre_sentence, len_tags, len_words)
	prob_unknown = pos_tagger.generate_unknown_prob(count_observation,len_tags)
	sentence = ""
	while True:
		sentence = input("Enter a sentence (or \'-exit\' to exit program): ")
		if sentence == '-exit':
			break
		get_viterbi_result(preprocess_input(sentence), len_tags, transition_matrix, observation_matrix, prob_unknown)

warnings.filterwarnings("ignore")

tag_list = defaultdict()
word_list, tags, len_tags, len_words, pre_sentence = pos_tagger.read_file()
for key, value in tags.items():
	tag_list[value] = key
get_tag(pre_sentence, len_tags, len_words)