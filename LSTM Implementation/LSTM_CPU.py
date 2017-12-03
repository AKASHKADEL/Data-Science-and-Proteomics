
# coding: utf-8

# # LSTM MODEL

# In[12]:

import data_processing as dp
import numpy as np
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


# ## Load Data and Create Train/Dev/Test Sets

# ### Human sequences

# In[2]:

human_sequences,_ = dp.load_FASTA('data/human_sequences.fasta')
human_train_idx, human_valid_idx, human_test_idx, human_train_labels,human_valid_labels, human_test_labels, human_GO_terms     = dp.load_test_sets('data/human_annotations_temporal_holdout.mat')

# Create train, validation, and test sets from the full list of human proteins
human_train_sequences = [human_sequences[i] for i in human_train_idx]
human_valid_sequences = [human_sequences[i] for i in human_valid_idx]
human_test_sequences = [human_sequences[i] for i in human_test_idx]

# Convert corresponding labels for train, validation, and test sets 
# from the full list of human proteins.
human_train_labels = torch.from_numpy(human_train_labels).type(torch.LongTensor)
human_valid_labels = torch.from_numpy(human_valid_labels).type(torch.LongTensor)
human_test_labels = torch.from_numpy(human_test_labels).type(torch.LongTensor)

# Create lengths for sequence representation averaging in FastText
human_train_lengths = dp.sequence_lengths(human_train_sequences)
human_valid_lengths = dp.sequence_lengths(human_valid_sequences)
human_test_lengths = dp.sequence_lengths(human_test_sequences)

# Convert protein sequence strings into long tensors where each int corresponds
# to one of 22 amino acids.  The length to truncate to is included.
human_train_tensors = dp.TransformAAsToTensor(human_train_sequences)
human_valid_tensors = dp.TransformAAsToTensor(human_valid_sequences)
human_test_tensors = dp.TransformAAsToTensor(human_test_sequences)


# #### Yeast sequences

# In[3]:

# Load yeast sequences and training data
yeast_sequences, yeast_protein_names = dp.load_FASTA('data/yeast_sequences.fasta')
yeast_train_idx, yeast_valid_idx, yeast_test_idx, yeast_train_labels, yeast_valid_labels,     yeast_test_labels, yeast_GO_terms = dp.load_test_sets('data/yeast_MF_temporal_holdout.mat')

# Create train, validation, and test sets from the full list of yeast proteins
yeast_train_sequences = [yeast_sequences[i] for i in yeast_train_idx]
yeast_valid_sequences = [yeast_sequences[i] for i in yeast_valid_idx]
yeast_test_sequences = [yeast_sequences[i] for i in yeast_test_idx]

# Convert corresponding labels for train, validation, and test sets from the full list of yeast proteins.
yeast_train_labels = torch.from_numpy(yeast_train_labels).type(torch.LongTensor)
yeast_valid_labels = torch.from_numpy(yeast_valid_labels).type(torch.LongTensor)
yeast_test_labels = torch.from_numpy(yeast_test_labels).type(torch.LongTensor)

# Create lengths for sequence representation averaging in FastText
yeast_train_lengths = dp.sequence_lengths(yeast_train_sequences)
yeast_valid_lengths = dp.sequence_lengths(yeast_valid_sequences)
yeast_test_lengths = dp.sequence_lengths(yeast_test_sequences)

# Convert protein sequence strings into long tensors where each int corresponds
# to one of 22 amino acids.  The length to truncate to is included.
yeast_train_tensors = dp.TransformAAsToTensor(yeast_train_sequences,500)
yeast_valid_tensors = dp.TransformAAsToTensor(yeast_valid_sequences,500)
yeast_test_tensors = dp.TransformAAsToTensor(yeast_test_sequences,500)


# ## Model

# ### LSTM class:  

# In[4]:

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_labels, batch_size):
        super(LSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)        
        self.embedding_size = emb_dim
        self.hidden_size = hidden_size
        self.output_size = num_labels
        self.batch_size = batch_size
        
        self.linear_f = nn.Linear(embedding_dim + hidden_size, hidden_size)
        self.linear_i = nn.Linear(embedding_dim + hidden_size, hidden_size)
        self.linear_ctilde = nn.Linear(embedding_dim + hidden_size, hidden_size)
        self.linear_o = nn.Linear(embedding_dim + hidden_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, num_labels)
        
        self.init_weights()
    
    def forward(self, data, hidden, c):
        """
        @param data: matrix of size (batch_size, max_sentence_length). Each row in data represents a 
            review that is represented using n-gram index. Note that they are padded to have same length.
        @param length: an int tensor of size (batch_size), which represents the non-trivial (excludes padding)
            length of each sentences in the data.
        """
        emb = self.embed(data)
        embs = torch.chunk(emb, emb.size()[1], 1)
        
        def step(emb, hid, c_t):
            combined = torch.cat((hid,emb),1)
            f = F.sigmoid(self.linear_f(combined))
            i = F.sigmoid(self.linear_i(combined))
            c_tilde = F.tanh(self.linear_ctilde(combined))
            c_t = f*c_t + i*c_tilde
            o = F.sigmoid(self.linear_o(combined))
            hid = o * F.tanh(c_t)
            return hid, c_t
        
        for i in range(len(embs)):
            hidden, c = step(embs[i].squeeze(), hidden, c)
        
        output = self.decoder(hidden)
        return output, hidden
    
    def init_hidden(self):
        h0 = Variable(torch.zeros(self.batch_size, self.hidden_size))
        c0 = Variable(torch.zeros(self.batch_size, self.hidden_size))
        return h0, c0
        
    def init_weights(self):
        initrange = 0.1
        lin_layers = [self.linear_f, self.linear_i, self.linear_ctilde, self.linear_o]
        em_layer = [self.embed]
     
        for layer in lin_layers+em_layer:
            layer.weight.data.uniform_(-initrange, initrange)
            if layer in lin_layers:
                layer.bias.data.fill_(0)
                


# ### Early stop condition

# In[5]:

def early_stop(val_loss_history, t=10, required_progress=0.001):
    """
    Stop the training if there is no non-trivial progress in k steps
    @param val_acc_history: a list contains all the historical validation acc
    @param required_progress: the next acc should be higher than the previous by 
        at least required_progress amount to be non-trivial
    @param t: number of training steps 
    @return: a boolean indicates if the model should earily stop
    """    
    cnt = 0 # initialize the count --> to store count of cases where difference in
                                    #  accuracy is less than required progress.
    
    if(len(val_loss_history) > 0): # if list has size > 0 
        for i in range(t): # start the loop
            index = len(val_loss_history) - (i+1) # start from the last term in list and move to the left
            if (index >= 1): # to check if index != 0 --> else we can't compare to previous value
                if ((val_loss_history[index-1] - val_loss_history[index]) < required_progress):
                    cnt += 1 # increase the count value
                else:
                    break # break if difference is greater 
    
    if(cnt != t): # if count is equal to t, return True
        return False
    else:
        return True


# ### Training loop

# In[9]:

def train_test(valid_sequences,valid_label,num_epochs,optimizer,data_iter,               model,training_length):
    losses = []
    total_batches = int(training_length/batch_size)
    validation_loss_history = []
        
    for epoch in range(1,num_epochs+1):
        stop_training = False
        for i, (train_data, train_labels,_) in enumerate(data_iter):
            model.train()
            model.zero_grad()

            hidden, c_t = model.init_hidden()
            outputs, hidden = model(train_data, hidden, c_t)
        
            loss = criterion(outputs, train_labels.float())
            losses.append(loss.data[0])
            loss.backward()
            optimizer.step()          

            # check validation loss
            if (i+1) % 100 ==0:
                model.eval()
                hidden, c_t = model.init_hidden()
                val_outputs, hidden = model(Variable(valid_sequences), hidden, c_t)                
                val_loss = criterion(val_outputs.data, test_eval_batch[eval_step][1].float()).data.numpy()
                valid_loss_history.append(val_loss)
                print('Epoch: [{}/{}], Step[{}/{}], Train loss: {}, Validation Loss: {}'.format(epoch, num_epochs, i+1, total_batches, np.mean(losses),val_loss))
            
	    if (i+1) % total_batches == 0:
		torch.save(model.state_dict(), '{}Model{}Epoch'.format(type,epoch)) # Saves model after every epoch                


# ## Train and Evaluate Model

# ### Hyperparameters 

# In[10]:

learning_rate = 0.05
vocab_size = 23 # number words in the vocabulary base
emb_dim = 8 # dimension for n-gram embedding
hidden_dim=12
num_epochs = 50 # number epoch to train
batch_size = 100


# ### Human Results

# In[ ]:

data_size = len(human_train_tensors)
num_labels = human_GO_terms.shape[0] #147

lstm = LSTM(vocab_size, emb_dim, hidden_dim,num_labels,batch_size)
criterion = nn.MultiLabelSoftMarginLoss()  
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

data_iter = dp.batch_iter(batch_size, human_train_tensors,                           human_train_labels,human_train_lengths)

# Model Training
type='Human'
train_test(human_valid_tensors, human_valid_labels, num_epochs,optimizer, data_iter, lstm, data_size) 


# ### Yeast Results

# In[ ]:

data_size = len(yeast_train_tensors)
num_labels = yeast_GO_terms.shape[0] 

lstm = LSTM(vocab_size, emb_dim, hidden_dim,num_labels,batch_size)
criterion = nn.MultiLabelSoftMarginLoss()  
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate) 

data_iter = dp.batch_iter(batch_size, yeast_train_tensors,                           yeast_train_labels,yeast_train_lengths)

if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

# Model Training
type='Yeast'
train_test(yeast_valid_tensors, yeast_valid_labels, num_epochs,optimizer, data_iter, lstm, data_size)
