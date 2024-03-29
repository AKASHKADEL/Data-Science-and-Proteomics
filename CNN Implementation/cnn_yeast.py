# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scipy.io as sio
from Bio import SeqIO
import collections
import sys

import torch
import argparse
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import random

#sys.path.append("..")
from evaluation_metrics import *
from data_processing import *


######## File params ########

parser = argparse.ArgumentParser()
parser.add_argument('--USE_CUDA', action='store_true', help='IF USE CUDA (Default == False)')
parser.add_argument('--hidden_size', type=int, default=100, help='Size of hidden layer')
parser.add_argument('--emb_dim', type=int, default=100, help='Embedding dimensions')
parser.add_argument('--n_epochs', type=int, default=10, help='Number of single iterations through the data')
parser.add_argument('--batch_size', type=int, default=50, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate (for both, encoder and decoder)')
parser.add_argument('--n_layers', type=int, default=1, help='Number of layers (for both, encoder and decoder)')
parser.add_argument('--eval_every', type=int, default=10, help='percentage of num iterations before evaluation')
parser.add_argument('--is_bidirectional', type=bool, default=False, help='Whether or not RNN is bidirectional')
parser.add_argument('--print_every', type=int, default=10, help='percentage of num iterations before printing')
parser.add_argument('--dropout_dec_p', type=float, default=0.1, help='Dropout (%) in the decoder')
parser.add_argument('--main_data_dir', type=str, default= "/scratch/ak6201/Capstone/data/", help='Directory where data is saved (in folders tain/dev/test)')
parser.add_argument('--out_dir', type=str, default="", help="Directory to save the models state dict (No default)")
parser.add_argument('--k', type=int, default=1, help="k for k-mers")
parser.add_argument('--l2_penalty', type=float, default=0.0005)
parser.add_argument('--save_every', type=int, default="20", help="Num of iterations required to save the model")
opt = parser.parse_args()
print(opt)

#Yeast sequences
fasta = opt.main_data_dir + 'yeast_sequences.fasta'
test_set_file = opt.main_data_dir + 'yeast_MF_temporal_holdout.mat'

sequences, names = load_FASTA(fasta)
train_inds, valid_inds, test_inds, y_trainYeast, y_validYeast, y_testYeast, go_termsYeast = load_test_sets(test_set_file)

train_seqsYeast = [sequences[i] for i in train_inds]
print('Number of training prots: ' + str(len(train_seqsYeast)))
valid_seqsYeast = [sequences[i] for i in valid_inds]
print('Number of validation prots: ' + str(len(valid_seqsYeast)))
test_seqsYeast = [sequences[i] for i in test_inds]
print('Number of testing prots: ' + str(len(test_seqsYeast)))


##########################
### 1. Data formation  ###
##########################

yTrainYeast = torch.from_numpy(y_trainYeast).type(torch.LongTensor)
yValidYeast = torch.from_numpy(y_validYeast).type(torch.LongTensor)
yTestYeast = torch.from_numpy(y_testYeast).type(torch.LongTensor)

train_seqsYeast_length = sequence_lengths_with_kmers(train_seqsYeast, opt.k)
valid_seqsYeast_length = sequence_lengths_with_kmers(valid_seqsYeast, opt.k)
test_seqsYeast_length = sequence_lengths_with_kmers(test_seqsYeast, opt.k)

####################
### 2.GET K-MERS ###
####################

if opt.k==1:
    k_mers_yeast = None
else:
    k_mers_yeast = get_k_mers(train_seqsYeast, valid_seqsYeast, test_seqsYeast, opt.k, org="yeast")

### GET TENSORS FOR THE DATA WITH KMERS

TrainSeqsYeast = TransformAAsToTensor_with_kmers(train_seqsYeast, opt.k, k_mers_yeast, acid_dict_yeast)
ValidSeqsYeast = TransformAAsToTensor_with_kmers(valid_seqsYeast, opt.k, k_mers_yeast, acid_dict_yeast)
TestSeqsYeast = TransformAAsToTensor_with_kmers(test_seqsYeast, opt.k, k_mers_yeast, acid_dict_yeast)

# Convert to cuda

if opt.USE_CUDA == True:
    TestSeqsYeast = TestSeqsYeast.cuda()
    yTestYeast = yTestYeast.cuda()


#############################
### 3. CNN IMPLEMENTATION ###
#############################

class CNN(nn.Module):
    """
    CNN model
    """
       
    def __init__(self, vocab_size, emb_dim, num_labels, hidden_size, n_layers=1, dropout=0.1, is_bidirectional = False):
       
        """
        @param vocab_size: size of the vocabulary. 
        @param emb_dim: size of the word embedding
        """
        super(CNN,self).__init__()
        Ci = 1
        Co = 2
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.conv13 = nn.Conv2d(Ci, Co, (3, emb_dim))
        self.conv14 = nn.Conv2d(Ci, Co, (4, emb_dim))
        self.conv15 = nn.Conv2d(Ci, Co, (5, emb_dim))
        self.convs = [self.conv13, self.conv14, self.conv15]
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(3*Co, num_labels) 
    
    def forward(self, x):
        x = self.embed(x) 
        x = x.unsqueeze(1) # (N,Ci,W,D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs] #[(N,Co,W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x) # (N,len(Ks)*Co)
        logit = self.fc1(x) # (N,C)
        #return logit
        return torch.nn.functional.softmax(logit)


        
#########################
### 4. Early stopping ###
#########################

def early_stop(val_acc_history, t=2, required_progress=0.00001):
    """
    Stop the training if there is no non-trivial progress in k steps
    @param val_acc_history: a list contains all the historical validation acc
    @param required_progress: the next acc should be higher than the previous by
        at least required_progress amount to be non-trivial
    @param t: number of training steps
    @return: a boolean indicates if the model should earily stop
    """
    # TODO: add your code here

    cnt = 0 # initialize the count --> to store count of cases where difference in
                                    #  accuracy is less than required progress.

    if(len(val_acc_history) > 0): # if list has size > 0
        for i in range(t): # start the loop
            index = len(val_acc_history) - (i+1) # start from the last term in list and move to the left
            if (index >= 1): # to check if index != 0 --> else we can't compare to previous value
                if (abs(val_acc_history[index] - val_acc_history[index-1]) < required_progress):
                    cnt += 1 # increase the count value
                else:
                    break # break if difference is greater

    if(cnt != t): # if count is equal to t, return True
        return False
    else:
        return True


#########################
### 5. Training Stage ###
#########################


def train(ValidSeqsYeast, yValidYeast, num_epochs, optimizer, data_iter, model, training_length):
    losses = []
    total_batches = int(training_length/ opt.batch_size) #375

    validate_every = int((opt.eval_every/100)*total_batches)
    show_every = int((opt.print_every/100)*total_batches)
    val_outputs = None
    eval_loss = None
    valid_label = None

    validation_losses = []

    for epoch in range(1, num_epochs+1):
        stop_training = False
        for i, (train_data, train_labels, length_batch) in enumerate(data_iter):
                                                # train_data size: (26, 34350) ; train_label size: (26, 147)
                                                # This needs to be modified. Max length is batch specific !!!!!
            model.train(True)
            model.zero_grad()
            outputs = model(train_data.cuda())
            loss = criterion(outputs, train_labels.cuda().float())
            losses.append(loss.data[0])
            loss.backward()


            clipped = torch.nn.utils.clip_grad_norm(model.parameters(), 0.5)
            # clip gradients because RNN
            for pr in model.parameters():
                pr.data.add_(-clipped, pr.grad.data)

            optimizer.step()

            if (i+1)%validate_every == 0:
                # Erly stop using validation loss
                valid_sequences, valid_label = reduced_set(ValidSeqsYeast, valid_seqsYeast_length, yValidYeast, 100)

                model.eval()
                val_outputs = model(Variable((valid_sequences).type(torch.LongTensor).cuda(), volatile=True))
                eval_loss = criterion(val_outputs.cpu().data, valid_label.float())
                print(eval_loss.data[0])
                validation_losses.append(eval_loss.data[0])
                stop_training = early_stop(validation_losses, 5)

            # Print statements
            if stop_training:
                print("earily stop triggered")
                break
            if (i+1) % show_every == 0:
                print('Epoch: [{0}/{1}], Step: [{2}/{3}], Train loss: {4}, Validation loss:{5}'.format(
                           epoch, num_epochs, i+1, total_batches, np.mean(losses)/(total_batches*epoch), np.mean(np.array(validation_losses))))
                
        evaluate_and_save(model, opt.out_dir, val_outputs, valid_label, losses, eval_loss.data[0], "yeast", epoch)
        if stop_training == True:
            break

        #torch.save(model.state_dict(), "{}/saved_model_yeast_{}.pth".format(opt.out_dir, (epoch+1)))



#############################
### 6. Learning the model ###
#############################


### YEAST ###

data_size = len(train_seqsYeast) 
num_labels = go_termsYeast.shape[0] 
if opt.k == 1:
    vocab_size = len(acid_dict_yeast) + 1
else:
    vocab_size = max(list(k_mers_yeast.values())) + 1

model = CNN(vocab_size, opt.emb_dim, num_labels, opt.hidden_size, n_layers=opt.n_layers, dropout=opt.dropout_dec_p, is_bidirectional=opt.is_bidirectional)

criterion = nn.MultiLabelSoftMarginLoss()

if opt.USE_CUDA == True:
	model = model.cuda()
	criterion = criterion.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.l2_penalty)

data_iter = batch_iter(opt.batch_size, TrainSeqsYeast, yTrainYeast, train_seqsYeast_length)
# Model Training
train(ValidSeqsYeast, yValidYeast, opt.n_epochs, optimizer, data_iter, model, data_size)

#torch.save(model.state_dict(), "{}/saved_model_yeast.pth".format(opt.out_dir))
