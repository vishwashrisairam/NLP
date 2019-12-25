import time
# import required modules
import numpy as np
import pandas as pd
import codecs
import itertools
import nltk 
nltk.download('wordnet')
from nltk.corpus import wordnet as wn 

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report

def extract_features(mode,filename): 
    feature_list=[]
    feature_labels=[]
    print('Enter extract features')
    with codecs.open(filename,'r',encoding='utf-8',errors='ignore') as doc:
        st=doc.read()
    _lines=st.splitlines()
    lines= list(filter(lambda x: x not in ['','. . O O','-DOCSTART- -X- -X- O'],_lines[1:]))

    print('Num of tags:',len(lines))
    for l in lines:
        fl=[]
        tag,pos,g_synhead,g_ner=l.split(' ')
        fl.append(tag)
        if pos not in pos_dict:
            pos_dict[pos]=len(pos_dict)+1
        # print(tag,pos,g_synhead,g_ner)
        lemmas = [i.lemmas() for i in wn.synsets(tag)]
        #lemmas
        lemmas=list(itertools.chain.from_iterable(lemmas))
        lemma_names=list(set([i.name() for i in lemmas]))
        fl.extend(lemma_names)

        #antonyms
        antonyms=[l.antonyms() for l in lemmas]
        antonyms=list(itertools.chain.from_iterable(antonyms))
        antonyms=list(set([i.name() for i in antonyms]))
        fl.extend(antonyms)

        #hypernyms
        hypernyms=[l.hypernyms() for l in wn.synsets(tag)]
        hypernyms=list(itertools.chain.from_iterable(hypernyms))
        hypernyms=list(set([l.name().split('.')[0] for l in hypernyms]))
        fl.extend(hypernyms)

        #hyponyms
        hyponyms=[l.hyponyms() for l in wn.synsets(tag)]
        hyponyms=list(itertools.chain.from_iterable(hyponyms))
        hyponyms=list(set([l.name().split('.')[0] for l in hyponyms]))
        fl.extend(hyponyms)

        #holonyms
        holonyms=[l.member_holonyms() for l in wn.synsets(tag)]
        holonyms=list(itertools.chain.from_iterable(holonyms))
        holonyms=list(set([l.name().split('.')[0] for l in holonyms]))
        fl.extend(holonyms)

        #meronyms
        meronyms=[l.part_meronyms() for l in wn.synsets(tag)]
        meronyms=list(itertools.chain.from_iterable(meronyms))
        meronyms=list(set([l.name().split('.')[0] for l in meronyms]))
        fl.extend(meronyms)
        
        fl.append(pos)

        # print('lemmas:',lemmas)
        # print('lemma_name:',lemma_names)
        # print('hypernyms:',hypernyms)
        # print('hyponyms:',hyponyms)
        # print('holonyms:',holonyms)
        # print('meronyms:',meronyms)
        # print('antonyms:',antonyms)

        if mode==1: # mode =1 for training
          for w in fl[:-1]:
              if w not in words_dict:
                  words_dict[w]=len(words_dict)+1
        
        feature_list.append(np.array(fl))

        if g_ner not in ner_tags_dict:
          ner_tags_dict[g_ner]=len(ner_tags_dict)+1

        feature_labels.append(g_ner)
    print('Exit extract features')
        
    return feature_list , feature_labels

def encode_features(features):
  # features=training_features[:10]
  print('Enter encode features')
  vocab_length=len(words_dict)
  pos_dict_length=len(pos_dict)

  encoded_features=[]
  for f in features :
    encf=np.zeros(vocab_length)
    for w in f[:-1]:
      if w in words_dict:
        ind=words_dict[w]-1
        encf[ind]+=1
    pos_ohv=np.zeros(pos_dict_length)
    pos_ohv[pos_dict[f[-1]]-1]+=1
    # print(pos_ohv)
    np.append(encf,pos_ohv)
    encoded_features.append(encf)  
  print('Exit encode features')
  return np.array(encoded_features)

def encode_labels(labels):
  # labels=training_labels[:10]
  ner_dict_length=len(ner_tags_dict)

  encoded_labels=[]
  for l in labels :
    # el=np.zeros(ner_dict_length)
    # el[ner_tags_dict[l]]+=1
    encoded_labels.append(ner_tags_dict[l])
  return np.array(encoded_labels)

def learn(encoded_features_training,encoded_labels_training,encoded_features_testing,encoded_labels_testing):
  clf=SGDClassifier(loss='log')

  classes=np.unique(encoded_labels_training).tolist()
  clf.partial_fit(encoded_features_training,encoded_labels_training,classes)

  predicted= clf.predict(encoded_features_testing)

  # print(ner_tags_dict)
  print(classification_report(encoded_labels_testing, predicted,labels=classes))

# Running the code 
start=time.time()

training_file='train.txt'
testing_file='test.txt'
pos_dict={}
words_dict={}
ner_tags_dict={}


#Preparation of training data
training_features,training_labels = extract_features(1,training_file)
# print('Words dict:',pos_dict)
# print('Words dict:',words_dict)
print('Dictionary length:',len(words_dict))


"""
encoded_features_training=encode_features(training_features[:50000])
encoded_labels_training = np.array(training_labels[:50000]) #encode_labels(training_labels[:10000])
print('Encoded features train:',encoded_features_training.shape)
print('Encoded Labels train:',encoded_labels_training.shape)
"""

batches=5000
num_tags=len(training_features)
s=0 
e=s+ batches 
num_iters= num_tags//batches
classes=list(ner_tags_dict.keys())
clf=SGDClassifier(loss='log')

for i in range(num_iters+1):
  print('Iteration: ',i )
  if i == num_iters:
    trf,trl=training_features[s:],training_labels[s:]
    break
  else:
    trf,trl=training_features[s:e],training_labels[s:e]
  s,e=s+batches,e+batches
  print(s,e)

  encoded_features_training=encode_features(trf)
  encoded_labels_training = np.array(trl)

  print('Training starts')
  clf.partial_fit(encoded_features_training,encoded_labels_training,classes)
  print('Training ends for the batch')

#Preparation of test data 
testing_features,testing_labels= extract_features(0,testing_file)
"""
encoded_features_testing=encode_features(testing_features[:50000])
encoded_labels_testing = np.array(testing_labels[:50000]) #encode_labels(testing_labels[:10000])
print('Encoded features test:',encoded_features_testing.shape)
print('Encoded Labels test:',encoded_labels_testing.shape)
"""

# learn(encoded_features_training,encoded_labels_training,encoded_features_testing,encoded_labels_testing)

# Testing 
batches=5000
num_tags,s,e=len(testing_features),0,batches
num_iters= num_tags//batches
encoded_features_testing=[]
encoded_labels_testing = np.array(testing_labels)

print(num_tags,num_iters,batches)

for i in range(num_iters+1):
  print('Iteration: ',i )
  if i == num_iters:
    tsf,tsl=testing_features[s:],testing_labels[s:]
    break
  else:
    tsf,tsl=testing_features[s:e],testing_labels[s:e]
  
  s,e=s+batches,e+batches
  print(s,e)

  tmp=encode_features(tsf)
  # print(tmp.shape)
  encoded_features_testing.extend(tmp)

  #Only for entire dataset #Memory Issues
  # predicted=clf.predict(tmp)

  # print('Classification Report:',classification_report(tsl, predicted,labels=classes))

print(np.array(encoded_features_testing).shape)
print(np.array(encoded_labels_testing).shape)

predicted= clf.predict(encoded_features_testing)

# print(ner_tags_dict)
print(classification_report(encoded_labels_testing, predicted,labels=classes))


end=time.time()
print('Time taken:', end-start)
