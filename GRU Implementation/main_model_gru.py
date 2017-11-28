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
parser.add_argument('--use_cuda', action='store_true', help='IF USE CUDA (Default == False)')
parser.add_argument('--hidden_size', type=int, default=100, help='Size of hidden layer')
parser.add_argument('--emb_dim', type=int, default=100, help='Embedding dimensions')
parser.add_argument('--n_epochs', type=int, default=10, help='Number of single iterations through the data')
parser.add_argument('--batch_size', type=int, default=80, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate (for both, encoder and decoder)')
parser.add_argument('--n_layers', type=int, default=1, help='Number of layers (for both, encoder and decoder)')
parser.add_argument('--eval_every', type=int, default=1, help='num iterations before evaluation')
parser.add_argument('--is_bidirectional', type=bool, default=False, help='Whether or not RNN is bidirectional')
parser.add_argument('--print_every', type=int, default=1, help='num iterations before printing')
parser.add_argument('--dropout_dec_p', type=float, default=0.1, help='Dropout (%) in the decoder')
parser.add_argument('--main_data_dir', type=str, default= "/scratch/ak6201/Capstone/data/", help='Directory where data is saved (in folders tain/dev/test)')
parser.add_argument('--out_dir', type=str, default="", help="Directory to save the models state dict (No default)")
opt = parser.parse_args()
print(opt)

#Human Sequences
fasta = opt.main_data_dir + 'human_sequences.fasta'
test_set_file = opt.main_data_dir + 'human_annotations_temporal_holdout.mat'

sequences, names = load_FASTA(fasta)
train_inds, valid_inds, test_inds, y_trainHuman, y_validHuman, y_testHuman, go_termsHuman = load_test_sets(test_set_file)

train_seqsHuman = [sequences[i] for i in train_inds]
print('Number of training prots: ' + str(len(train_seqsHuman)))
valid_seqsHuman = [sequences[i] for i in valid_inds]
print('Number of validation prots: ' + str(len(valid_seqsHuman)))
test_seqsHuman = [sequences[i] for i in test_inds]
print('Number of testing prots: ' + str(len(test_seqsHuman)))

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

yTrainHuman = torch.from_numpy(y_trainHuman).type(torch.LongTensor)
yValidHuman = torch.from_numpy(y_validHuman).type(torch.LongTensor)
yTestHuman = torch.from_numpy(y_testHuman).type(torch.LongTensor)

k = 2 #value for kmers

train_seqsHuman_length = sequence_lengths_with_kmers(train_seqsHuman, k)
valid_seqsHuman_length = sequence_lengths_with_kmers(valid_seqsHuman, k)
test_seqsHuman_length = sequence_lengths_with_kmers(test_seqsHuman, k)
train_seqsYeast_length = sequence_lengths_with_kmers(train_seqsYeast, k)
valid_seqsYeast_length = sequence_lengths_with_kmers(valid_seqsYeast, k)
test_seqsYeast_length = sequence_lengths_with_kmers(test_seqsYeast, k)

####################
### 2.GET K-MERS ###
####################

k_mers_human = get_k_mers(train_seqsHuman, valid_seqsHuman, test_seqsHuman, k)
k_mers_yeast = get_k_mers(train_seqsYeast, valid_seqsYeast, test_seqsYeast, k)

### GET TENSORS FOR THE DATA WITH KMERS

TrainSeqsYeast = TransformAAsToTensor_with_kmers(train_seqsYeast, k, k_mers_yeast)
ValidSeqsYeast = TransformAAsToTensor_with_kmers(valid_seqsYeast, k, k_mers_yeast)
TestSeqsYeast = TransformAAsToTensor_with_kmers(test_seqsYeast, k, k_mers_yeast)

# This can take a while for k >= 2
TrainSeqsHuman = TransformAAsToTensor_with_kmers(train_seqsHuman, k, k_mers_human)
ValidSeqsHuman = TransformAAsToTensor_with_kmers(valid_seqsHuman, k, k_mers_human)
TestSeqsHuman = TransformAAsToTensor_with_kmers(test_seqsHuman, k, k_mers_human)


#############################
### 3. GRU IMPLEMENTATION ###
#############################


class RNN_GRU(nn.Module):
    """
    GRU model
    """

    def __init__(self, vocab_size, emb_dim, num_labels, hidden_size, n_layers=1, dropout=0.1, is_bidirectional = False):

        """
        @param vocab_size: size of the vocabulary.
        @param emb_dim: size of the word embedding
        """
        super(RNN_GRU, self).__init__()

        self.num_labels = num_labels
        self.num_directions = 1 # it is 2 if the rnn is bidirectional
        self.hidden_size = hidden_size
        self.is_bidirectional = is_bidirectional
        self.dropout = nn.Dropout(p=dropout)
        self.embed = nn.Embedding(vocab_size+1, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden_size, n_layers, dropout, bidirectional=is_bidirectional)
        self.linear = nn.Linear(self.num_directions*hidden_size, num_labels)


    def forward(self, input_seqs):
        """
        @param data: matrix of size (batch_size, max_sentence_length). Each row in data represents a
            review that is represented using n-gram index. Note that they are padded to have same length.
        @param length: an int tensor of size (batch_size), which represents the non-trivial (excludes padding)
            length of each sentences in the data.
        """

        embedded = self.embed(input_seqs) # size = (max_length, batch_size, embed_size)
        hidden = None
        outputs, hidden = self.gru(embedded, hidden) # outputs are supposed to be probability distribution right?
        if self.is_bidirectional == True:
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs

        last_hidden = self.dropout(outputs[-1,:,:].squeeze())
        output_probability = torch.nn.functional.sigmoid(self.linear(last_hidden))

        return output_probability # size : (batch_size, num_labels)


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


def train(valid_sequences, valid_label, num_epochs, optimizer, data_iter, model, training_length, threshold):
    losses = []
    total_batches = int(training_length/ opt.batch_size) #375

    validation_losses = []
    F_scores = []
    calculated_f_score = np.zeros(len(threshold))
    max_precision = np.zeros(len(threshold))
    max_recall = np.zeros(len(threshold))

    for epoch in range(1, num_epochs+1):
        stop_training = False
        for i, (train_data, train_labels, length_batch) in enumerate(data_iter):
                                                # train_data size: (26, 34350) ; train_label size: (26, 147)
                                                # This needs to be modified. Max length is batch specific !!!!!
            model.train(True)
            model.zero_grad()
            outputs = model(train_data.transpose(0,1))
            loss = criterion(outputs, train_labels.float())
            losses.append(loss.data[0])
            loss.backward()


            clipped = torch.nn.utils.clip_grad_norm(model.parameters(), 0.5)
            # clip gradients because RNN
            for pr in model.parameters():
                pr.data.add_(-clipped, pr.grad.data)

            optimizer.step()

            if i%opt.eval_every == 0:
#                 # Erly stop using validation loss
                model.eval()
                val_outputs = model(Variable((valid_sequences).transpose(0,1).type(torch.LongTensor), volatile=True))
                eval_loss = criterion(val_outputs.data, valid_label.type(torch.FloatTensor))
                print(eval_loss.data[0])
                validation_losses.append(eval_loss.data[0])
                stop_training = early_stop(validation_losses, 3)

            # Print statements
            if stop_training:
                print("earily stop triggered")
                break
            if (i+1) % opt.print_every == 0:
#                 print('Epoch: [{0}/{1}], Step: [{2}/{3}], Train loss: {4}, F_Score: {5}, Validation loss:{6}'.format(
#                            epoch, num_epochs, i+1, total_batches, np.mean(losses)/(total_batches*epoch), np.max(calculated_f_score), np.mean(np.array(validation_acc_history)[:,-1])))
                print('Epoch: [{0}/{1}], Step: [{2}/{3}], Train loss: {4}, Validation loss:{5}'.format(
                           epoch, num_epochs, i+1, total_batches, np.mean(losses)/(total_batches*epoch), np.mean(np.array(validation_losses))))
        if stop_training == True:
            break

    #return calculated_f_score, max_precision, max_recall


#############################
### 6. Learning the model ###
#############################


### HUMAN ###

data_size = len(train_seqsHuman) #9751
num_labels = go_termsHuman.shape[0] #147
vocab_size = len(acid_dict) + len(k_mers_human)

model = RNN_GRU(vocab_size, 50, num_labels, opt.hidden_size, n_layers=opt.n_layers, dropout=opt.dropout_dec_p, is_bidirectional=opt.is_bidirectional)
criterion = nn.MultiLabelSoftMarginLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)

data_iter = batch_iter(opt.batch_size, TrainSeqsHuman, yTrainHuman, train_seqsHuman_length)
threshold = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1]
# Model Training
ValidSeqsHuman_small, yValidHuman_small = reduced_set(ValidSeqsHuman, valid_seqsHuman_length, yValidHuman, 100)
#f2, p2, r2 = train(ValidSeqsHuman_small, yValidHuman_small, num_epochs, optimizer, data_iter, model, data_size, threshold)
train(ValidSeqsHuman_small, yValidHuman_small, opt.n_epochs, optimizer, data_iter, model, data_size, threshold)
torch.save(model.state_dict(), "{}/saved_model_human_{}.pth".format(opt.out_dir))

### YEAST ###
