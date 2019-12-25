Additional Python packages to be installed apart from pytorch

pip install gensim

For gensim you might face some issues while importing the gensim module. Try the below workaround in case: 
1) Uninstall numpy and gensim(if it exists)
2) pip install numpy && pip install gensim


Instructions for running the progam:

1. Open the terminal in the directory of this project file.
2. To run the file enter the command:
   python test.py (** optional command line arguments) 

The program takes can take upto four command line arguments:
1) --model model_name : This defines the type of model to train. 
    Possible Values: baseline, rnn, lstm,gru,selfattention. Default is baseline 
2) --epochs (no_epochs) : This defines the number of epochs you want the model to run. 
      Default value is 5 
3) --learningrate : This defines the learning rate for the model. Default is 0.01
4) --bidirectional : This so only for the rnn models. Possible values 1 if bidirectional else 0 

Example : python test.py --model lstm --epochs 10 --learningrate 0.001 

 The output contains the following details:
  - The accuracy report 

Note: For the word-embedding part, the programs loads the vector and stores it in a .pkl file. 
The next time its required during training, it picks from the embedding dictionary directly from the .pkl file. 