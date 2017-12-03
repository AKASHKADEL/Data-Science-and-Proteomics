
# coding: utf-8

# # LSTM MODEL

import data_processing as dp
import evaluation_metrics as em
import numpy as np
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

output_file = 'lstm_gpu_output/yeast/baseline/'

# Set Hyper-parameters:
emb_dim = 250 # dimension for n-gram embedding
hidden_dim=750
L2_penalty = 0.0005

# ## Load Data and Create Train/Dev/Test Sets

# Load yeast sequences and training data
yeast_sequences, yeast_protein_names = dp.load_FASTA('data/yeast_sequences.fasta')
yeast_train_idx, yeast_valid_idx, yeast_test_idx, yeast_train_labels, yeast_valid_labels,    yeast_test_labels, yeast_GO_terms = dp.load_test_sets('data/yeast_MF_temporal_holdout.mat')

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

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_labels, batch_size):
        super(LSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)        
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
        embs = torch.chunk(emb.float().cuda(), emb.size()[1], 1)
        
        def step(emb, hid, c_t):
            c_t = c_t.float().cuda()
            hid = hid.float().cuda()
            emb = emb.float().cuda()
            combined = torch.cat([hid,emb],1).float().cuda()
            f = F.sigmoid(self.linear_f(combined)).float().cuda()
            i = F.sigmoid(self.linear_i(combined)).float().cuda()
            c_tilde = F.tanh(self.linear_ctilde(combined)).float().cuda()
            c_t = f*c_t + i*c_tilde
            o = F.sigmoid(self.linear_o(combined)).float().cuda()
            hid = o * F.tanh(c_t).float().cuda()
            return hid, c_t
        
        for i in range(len(embs)):
            hidden, c = step(embs[i].squeeze(), hidden, c)
        
        output = self.decoder(hidden)
        return F.sigmoid(output), hidden
    
    def init_hidden(self):
        h0 = Variable(torch.zeros(self.batch_size, self.hidden_size))
        c0 = Variable(torch.zeros(self.batch_size, self.hidden_size))
        return h0, c0
        
    def init_weights(self):
        initrange = 0.1
        lin_layers = [self.linear_f, self.linear_i, self.linear_ctilde, self.linear_o]
        em_layer = [self.embed]
     
        for layer in lin_layers+em_layer:
            layer.weight.data.uniform_(-initrange, initrange).cuda()
            if layer in lin_layers:
                layer.bias.data.fill_(0).cuda()
                


# ### Early stop condition

def early_stop(val_loss_history, t=2, required_progress=0.001):
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

def train_test(num_epochs,optimizer,train_iter,eval_batches,model,training_length,output_file):
    losses = []
    total_batches = int(training_length/batch_size)
    valid_loss_history = []
    epoch = 0
        
    while epoch <= num_epochs:
        for batch, (train_data, train_labels,_) in enumerate(train_iter):
            model.train()
            model.zero_grad()

            hidden, c_t = model.init_hidden()
            hidden = hidden.float().cuda()
            c_t = c_t.float().cuda()
            outputs, hidden = model(train_data.cuda(), hidden, c_t)
        
            loss = criterion(outputs, train_labels.float())
            losses.append(loss.data[0])
            loss.backward()
            optimizer.step()          

	    if batch % 10 == 0:
                model.eval()
                hidden, c_t = model.init_hidden()
                valid_sequences = Variable(eval_batches[0][0])
                valid_labels = eval_batches[0][1]
                val_outputs, hidden = model(valid_sequences.cuda(), hidden, c_t)
                val_loss = criterion(val_outputs.cpu().data.round(), valid_labels.float()).data[0]
                valid_loss_history.append(val_loss)
                print('Epoch:{}/{},Batch:{}, Avg Train Loss: {}, Val Loss: {}'.format(epoch, num_epochs, batch, np.mean(losses),val_loss))

	    if batch % total_batches == 0:
		predictions = val_outputs.cpu().data.round().numpy()
		actual = valid_labels.numpy()
		AUC = em.calculate_AUC(predictions,actual)[0]
		AUPR = em.calculate_AUPR(predictions,actual)
		# ADD F-SCORE
                torch.save(model.state_dict(), output_file+'%sEpoch%s AUC:%0.4f AUPR: %0.4f'%(organism,epoch,AUC,AUPR))  

                stop_training = early_stop(valid_loss_history)            
                if stop_training:
                    print('early stop triggered')
	            return 0
		epoch+=1

# ## Train and Evaluate Model

learning_rate = 0.001
num_epochs = 30 
batch_size = 100

# ### Yeast Results
vocab_size = 21 # number words in the vocabulary base
data_size = len(yeast_train_tensors)
num_labels = yeast_GO_terms.shape[0] 

lstm = LSTM(vocab_size, emb_dim, hidden_dim,num_labels,batch_size)
criterion = nn.MultiLabelSoftMarginLoss()  
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate, weight_decay=L2_penalty) 

data_iter = dp.batch_iter(batch_size, yeast_train_tensors,yeast_train_labels,yeast_train_lengths)
dev_batches = dp.eval_iter(batch_size,yeast_valid_tensors,yeast_valid_labels)

if torch.cuda.is_available():
    lstm = lstm.cuda()
    criterion = criterion.cuda()

# Model Training
organism='Yeast'
train_test(num_epochs,optimizer,data_iter,dev_batches,lstm,data_size,output_file)



