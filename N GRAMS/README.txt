This python program trains a language model on the corpus (provided in brown_corpus_reviews.txt ) and finds out which of the two sentences 
is more probable.


The sentences are: 
S1: Milstein is a gifted violinist who creates all sorts of sounds and arrangements .
S2: It was a strange and emotional thing to be at the opera on a Friday night .

Instructions for running the progam:

1. Open the terminal in the directory of this exercise file.
2. To run the file enter the command:
   python NGrams.py -N 2 -b 1 

   The program takes two command line arguments:
   - an integer N {2,3} that indicates wether to use bigram or trigram 
   - an integer b {0,1} that indicates wether the model should be trained with or without smoothing
 
 The output contains the following details:
  - A matrix showing N-Gram counts for each sentence
  - A matrix showing N-Gram probability for each sentence
  - The probability of each sentence as computed using the model

  The Output_Birgram.docx(or txt) and Output_Trigram.docx(or txt) files contain the output of the program run on the Bigram and Trigram cases with and without
  smoothing respectively.
