Python packages used 
nltk 
gensim

Installation instructions for the above packages 
pip install nltk 
pip install gensim

For gensim you might face some issues while importing the gensim module. Try the below workaround in case: 
1) Uninstall numpy and gensim(if it exists)
2) pip install numpy && pip install gensim

Download the wordnet data using the below command 
python 
import nltk 
nltk.download('wordnet')


Instructions for running the progam:

1. Open the terminal in the directory of this project file.
2. To run the file enter the command:
   
   python model.py  -> For Logistic Regression Model 
   
   python ner_lstm.py -> For RNN LSTM Model


The output contains the following details:
  - The accuracy of the classification and the precision, recall and F-Scores for each tag  
