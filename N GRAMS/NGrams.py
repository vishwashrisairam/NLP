# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 00:16:56 2019

@author: Vishwashrisairam
"""

import sys
# f1=open("output.txt","a")

def getBigramProbability(sent,corpus,sv):
    print("----Bigram----")
    arr=list(set(sent.split()))
    arr=[word.lower() for word in arr]
    arr_len=len(arr)
    v=len(set(corpus))
    print('Num of words in corpus',v)
    
    #initialize the matrix
    matrix=[[0 for i in range(arr_len)] for j in range(arr_len)]

    #counting the word starts
    arr_count=[]
    for word in arr:
        arr_count.append(corpus.count(word))

    for i in range(len(arr)):
        for j in range(len(arr)):
            count=getBigramCorpusCount(arr[i],arr[j],corpus)
            if sv=='0':
                matrix[i][j]=count
            else:
                #matrix[i][j]=count+1
                matrix[i][j]=(count+1)*(arr_count[i]/(arr_count[i]+v))

    #print matrix in readable format
    print("-----------------Count matrix---------------")
    print_matrix(arr,matrix)
    #calculate probability
    for i,row in enumerate(matrix):
        for j,val in enumerate(row):
            if sv=='0':
                val=int(val)/arr_count[j]
            else:
                val=(int(val)+1)/(arr_count[j]+v)
            matrix[i][j]=val
    print("-----------------Probability matrix---------------")
    #print(matrix)
    print_matrix(arr,matrix)

    #find probability of the sentence
    prob=1
    eos_marker_index=arr.index('.') # Assumption '.' acts as maker for both begin and end of a sentence
    words=sent.split()
    words=[w.lower() for w in words]
    for i,w in enumerate(words):
        # print(i,w)
        word_index=arr.index(w)
        if i==0: # For first word in the sentense, assuming '.' as end of previous sentence and hence finding the probability of (.,word) bigram
            prob=prob*matrix[eos_marker_index][word_index]
        else:
            prev_word_index=arr.index(words[i-1])
            prob=prob*matrix[prev_word_index][word_index]

    return prob

def getBigramCorpusCount(a,b,corpus):
    count =0
    for i,w in enumerate(corpus):
        if w==a and i-1>0 and corpus[i-1]==b:
            count=count+1
    return count

def getTrigramCorpusCount(a,b,c,corpus):
    count =0
    for i,w in enumerate(corpus):
        if w==a and i-1>0 and corpus[i-1]==b and i-2>0 and corpus[i-2]==c:
            count=count+1
    return count

def getTrigramProbability(sent,corpus,sv):
    print("----Trigram----")
    arr=list(set(sent.split()))
    arr_len=len(arr)
    v=len(set(corpus))
    print('Num of words in corpus',v)
    
    #initialize the matrix
    matrix={}
    for i in range(arr_len):
        for j in range(arr_len):
            count=getBigramCorpusCount(arr[i],arr[j],corpus)
            matrix[(arr[i],arr[j],count)]=[]
            count=getBigramCorpusCount(arr[j],arr[i],corpus)
            matrix[(arr[i],arr[j],count)]=[]
    
    #counting the word starts
 
    for i in range(len(arr)):
        for j in matrix.keys():
            count=getTrigramCorpusCount(arr[i],j[1],j[0],corpus)
            if sv=='0':
                matrix[j].append(count)
            else:
                matrix[j].append((count+1)*(j[2]/(j[2]+v)))

    #print matrix in readable format
    # print(matrix)
    print("-----------------Count matrix---------------")
    print_trigram_matrix(arr,matrix)
    
    #calculate probability
    for j in matrix.keys():
        for i in range(len(matrix[j])):
            if sv=='0': # Smoothing Value = 0
                if j[2]!=0:
                    matrix[j][i]=(matrix[j][i]/j[2])
                else:
                    matrix[j][i]=0  # For the case when the bigram count is zero 
            else:
                matrix[j][i]=(matrix[j][i]+1)/(j[2]+v)
    print("-----------------Probability matrix---------------")
    # print(matrix)
    print_trigram_matrix(arr,matrix)

    #find probability of the sentence
    prob=1
    words=sent.split()
    words=[w.lower() for w in words]
    for i,w in enumerate(words):
        # for the first two words considering the bigram probability since we can't take occurence prob for previous two words
        if i==0 :
            # prob=prob*matrix[word_index][eos_marker_index]
            prob=prob*(getBigramCorpusCount(w,'.',corpus)/corpus.count('.')) # Assumption: '.' is the maker for both begin and end of sentence 
        elif i==1:
            prob=prob*(getBigramCorpusCount(words[i],words[i-1],corpus)/corpus.count(words[i-1]))
        else:
            prev_word1=words[i-1]
            prev_word2=words[i-2]
            for j in matrix.keys():
                if j[0]==prev_word2 and j[1]==prev_word1:
                    prob=prob*(matrix[j][i])
                    break

    return prob

def print_matrix(arr,matrix):
    for i in range(len(arr)+1):
        for j in range(len(arr)+1):
            if i==0 and j==0:
                print(' ',end="\t")
            elif i==0:
                print(arr[j-1],end="\t")
            elif j==0:
                print(arr[i-1],end="\t")
            else:
                print(matrix[i-1][j-1],end="\t")
        print("\n")   

def print_trigram_matrix(arr,matrix):
    for i in range(len(arr)+1):
        if i==0:
            print('',end='\t')
        else:
            print(arr[i-1],end='\t')
    print('\n')
    
    for j in matrix.keys():
        print("({},{})".format(j[0],j[1]),end="\t")
        for i in matrix[j]:
            print(i,end="\t")
        print('\n')

def main(argv):
    # Default is bigram without smoothing
    if len(argv)==0:
        ngram_val=2
        smoothing_val=0
    else:
        ngram_val=argv[1]
        smoothing_val=argv[3]
    # print(ngram_val,smoothing_val)

    f=open("brown_corpus_reviews.txt","r")
    corpus=f.read().split()
    corpus= [word.lower() for word in corpus]


    S1="Milstein is a gifted violinist who creates all sorts of sounds and arrangements ."
    S2="It was a strange and emotional thing to be at the opera on a Friday night ."
    

    prb1=1
    prb2=1

    if ngram_val=='2':
        if str(smoothing_val)=='1':
            print("***************************Bigram with smoothing**********************")
        elif str(smoothing_val)=='0':
            print("***************************Bigram without smoothing**********************")
        else:
            print('Please enter valid smoothing values. It can be 0 for no smoothing or 1 for add one smoothing.')
            sys.exit()
        print("\nSentence 1:",S1)
        prb1=getBigramProbability(S1,corpus,smoothing_val)
        print("\nSentence 2:",S2)
        prb2=getBigramProbability(S2,corpus,smoothing_val)
    elif ngram_val=='3':
        if str(smoothing_val)=='1':
            print("***************************Trigram with smoothing**********************")
        elif str(smoothing_val)=='0':
            print("***************************Trigram without smoothing**********************")
        else:
            print('Please enter valid smoothing values. It can be 0 for no smoothing or 1 for add one smoothing.')
            sys.exit()
        print("\nSentence 1:",S1)
        prb1=getTrigramProbability(S1,corpus,smoothing_val)
        print("\nSentence 2:",S2)
        prb2=getTrigramProbability(S2,corpus,smoothing_val)
    else:
        print("Please enter valid values of N. It can be 2 for bigram and 3 for Trigram")
        sys.exit(0)

    print("\nProbability of sentence 1",prb1)
    print("\nProbability of sentence 2",prb2)
    if prb1> prb2:
        print("Sentence 1 is more likely to occur.")
    elif prb1< prb2:
        print("Sentence 2 is more likely to occur.")
    else:
        if prb1==0:
            print("Both sentences are equally unlikely to occur.")
        else:
            print("Both sentences are equally likely to occur.")
    
    f.close()
    #f1.close()

if __name__ == "__main__":
    main(sys.argv[1:])
