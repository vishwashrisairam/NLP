import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch import optim
import gensim
import numpy as np
from sklearn.metrics import classification_report
import codecs
import os
import pickle

def load_data(filename):
    feature_list=[]
    feature_labels=[]
    pos_list = []
    vocab ={}
    ner_dic ={}

    with codecs.open(filename,'r',encoding='utf-8',errors='ignore') as doc:
        st=doc.read()
    _lines=st.splitlines()
    lines= list(filter(lambda x: x!='',_lines[1:]))

    print('Num of tags:',len(lines))
    for l in lines[:9600]:
        temp_list = []
        tag,pos,g_synhead,g_ner=l.split(' ')
        if tag not in vocab:
            vocab[tag] = len(vocab)+1
        if g_ner not in ner_dic:
            ner_dic[g_ner] = len(ner_dic)+1
        feature_list.append(tag)
        feature_labels.append(g_ner)

    return feature_list, feature_labels, vocab, ner_dic

def encode_features(vocab, train_features):
    temp = []
    for i in train_features:
        if i in vocab:
            temp.append(vocab[i])
        else:
            temp.append(0)
    return temp


def load_embedding_file(embedding_file, token_dict):
    embed = {}
    if not os.path.isfile(embedding_file):
        sys.exit("Input embedding path is not a file")
    if type(token_dict) is not dict:
        sys.exit("Input a dictionary!")
    model = gensim.models.KeyedVectors.load_word2vec_format(embedding_file)
    for i,j in token_dict.items():
        if i in model.vocab:
            embed[j] = torch.from_numpy(model[i])
        else:
            embed[j] = torch.zeros(300)
    with open('dictionary.p','wb') as fp:
        pickle.dump(embed,fp)

def embed_dict():
    with open('dictionary.p','rb') as f:
        embed_dictionary = pickle.load(f)
    embed_dictionary[0] = torch.zeros(300)
    return embed_dictionary

def embedding(embed_dictionary,input_words):
    res = np.zeros(shape =(32,1,300))
    for j,i in enumerate(input_words):
        if bacov[int(i)] in embed_dictionary:
            res[j,0,:] = embed_dictionary[bacov[int(i)]]
    res = torch.from_numpy(res).double()
    return res

def create_data_loader(encoded_features, test_features, labels, test_labels, batch_size = 32):
 
    if type(encoded_features) is not list or type(labels) is not list:
        sys.exit("Please provide a list to the method")

    train_data = TensorDataset(torch.FloatTensor(encoded_features), torch.FloatTensor(labels))
    valid_data = TensorDataset(torch.FloatTensor(test_features[:len(test_features)//2]), torch.FloatTensor(test_labels[:len(test_features)//2]))
    test_data = TensorDataset(torch.FloatTensor(test_features[len(test_features)//2:]), torch.FloatTensor(test_labels[len(test_features)//2:]))

    train_loader = DataLoader(train_data,  batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data,  batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data,  batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader


class LSTM(nn.Module):
    def __init__(self, vocab_size, output_size, hidden_dim, embedding_dim=300,n_layers=2, drop_prob=0.5):

        super(LSTM, self).__init__()

        self.e_dict = embed_dict()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,dropout = drop_prob, batch_first=True, bidirectional=True)

        # dropout layer
        self.dropout = nn.Dropout(0.3)

        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.soft = nn.LogSoftmax(dim=2)


    def forward(self, x, hidden):
        #batch_size = x.size(0)

        # embeddings
        embeds = embedding(self.e_dict,x)

        #rnn model
        out, hidden = self.lstm(embeds, hidden)

        # stack up outputs
        out = out.contiguous().view(-1, self.hidden_dim)

        # dropout and fully-connected layer
        out = self.dropout(out)
        out = self.fc(out)
        
        out = out.view(batch_size, -1)

        return out


    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers*2,batch_size,self.hidden_dim).zero_(),weight.new(self.n_layers*2,batch_size, self.hidden_dim).zero_())
        return hidden


def train(model,train_loader,valid_loader, test_loader,criterion,epochs=20):
    batch_size = 32
    counter = 0
    print_every = 100
    model.double()
    model.train()
    # train for some number of epochs
    for e in range(epochs):
        # initialize hidden state
        h = model.init_hidden(batch_size)
        # batch loop
        for inputs, labels in train_loader:
            counter += 1

            h = tuple([each.data for each in h])

            # zero accumulated gradients
            model.zero_grad()
            output = model(inputs.long(), h)
            # calculate the loss and perform backprop
            loss = criterion(output.squeeze(), labels.long())
            loss.backward()
           
            optimizer.step()

            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_h = model.init_hidden(batch_size)
                val_losses = []
                model.eval()
                for inputs, labels in valid_loader:

                    val_h = tuple([each.data for each in val_h])
                    output = model(inputs.long(), val_h)

                    val_loss = criterion(output.squeeze(), labels.long())

                    val_losses.append(val_loss.item())

                model.train()
                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()))


# Testing

def test(model, test_loader, criterion):
    batch_size = 32
    test_losses = []
    test_pred = []
    test_labels = []
    num_correct = 0

    # init hidden state
    h = model.init_hidden(batch_size)
    model.double()
    model.eval()
    # iterate over test data
    for inputs, labels in test_loader:
        h = tuple([each.data for each in h])
        # get predicted outputs
        output = model(inputs.long(), h)
        output = output[0,:32]
        output_arr=output.cpu().detach().numpy()
        output_arr=np.ceil(output_arr)
        output_arr=list(np.ravel(output_arr))
        # max_ind=output_arr.index(max(output))
        label_arr=labels.long().cpu().detach().numpy()
        label_arr=list(np.ravel(label_arr))
        
        test_pred.extend(output_arr)
        test_labels.extend(label_arr)

    a=[]
    b=[]
    for i,j in zip(test_pred, test_labels):
        a.append(i)
        b.append(j)
    print(classification_report(y_pred=a, y_true=b))



train_features, train_labels, vocab, ner_dic = load_data('train.txt')
test_features, test_labels, vocab1, ner_dic1 = load_data('test.txt')
bacov={}
for k in vocab.keys():
    bacov[vocab[k]] = k
bacov[0] = 0
#print(bacov)
train_encoded_features = encode_features(vocab, train_features)
test_encoded_features = encode_features(vocab, test_features)
train_encoded_labels = encode_features(ner_dic, train_labels)
test_encoded_labels = encode_features(ner_dic, test_labels)
load_embedding_file('wiki-news-300d-1M.vec', vocab)
#embed_dict()

output_size = 32
embedding_dim = 300
hidden_dim = 256
batch_size = 32
train_loader,valid_loader,test_loader = create_data_loader(train_encoded_features, test_encoded_features, train_encoded_labels, test_encoded_labels)
vocab_size = len(vocab)+1
learn_rate = 0.1

model = LSTM(vocab_size, output_size, hidden_dim)
print(model)


# Define loss and optimizer here
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learn_rate)


train(model,train_loader,valid_loader, test_loader, criterion)

test(model, test_loader, criterion)
