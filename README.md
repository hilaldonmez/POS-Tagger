# POS-Tagger
Part-of-Speech (POS) of a word is the category of the word with respect to its grammatical properties in a natural language. In natural language processing (NLP), determining the POS of a word is one of the base steps for further processing functions because knowing POS tag of a word is an important information during ambiguity resolution. POS tagging is used in NLP sub-tasks like text summarization and named entity recognition.

Below are Python 3 packages needed to run the code:
* numpy
* pandas
* scikit-learn

After setting the environment, our Part-of-Speech tagger can be used in two ways:
1. Evaluation of system by splitting input data to training and test sets then calculating the accuracy.
     $ python3 evaluate.py  <br /> 
### Possible parameters: 
* -batch: The number of iterations to go over the code in order to increase the randomness. Default is 10.
* -cross-val-batch: The number of slices to be used during splitting of corpus to train and test data. Default is 6.      
     $ python3 evaluate.py -cross-val-batch 2
* -morph: To use morphological information instead of Hapax Legonema during handling of unknown words.  <br /> 
     $ python3 evaluate.py -morph
* -stem: To use the stemmed versions of tokens during processing. <br />
     $ python3 evaluate.py -stem
     
2. Taking an input in natural language and determining the most likely Part-of-Speech tag of every token. <br />
     $ python3 tag.py  <br />

After starting the program, text input is taken. In order to terminate the code -exit is typed.

### Program Structure
* Preprocessing data
* Creating observation and transition matrices
* Implementing Viterbi Algorithm
* Smoothing
* Solving Unknown Word Problem with Hapax Legomena and Morphological Rules

