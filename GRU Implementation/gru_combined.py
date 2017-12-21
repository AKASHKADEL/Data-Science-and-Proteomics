# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scipy.io as sio
from Bio import SeqIO
import collections
import sys
import pandas as pd
import data_processing as dp

import torch
import argparse
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import random

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
parser.add_argument('--k', type=int, default="1", help="k for k-mers")
parser.add_argument('--l2_penalty', type=float, default=0.0005)
parser.add_argument('--save_every', type=int, default="20", help="Num of iterations required to save the model")
opt = parser.parse_args()
print(opt)



human_sequences,_ = dp.load_FASTA(opt.main_data_dir+'human_sequences.fasta')
human_train_idx, human_valid_idx, human_test_idx, human_train_labels,human_valid_labels, human_test_labels, human_GO_terms     = dp.load_test_sets(opt.main_data_dir+'human_annotations_temporal_holdout.mat')

# Create train, validation, and test sets from the full list of human proteins
human_train_sequences = [human_sequences[i] for i in human_train_idx]
human_valid_sequences = [human_sequences[i] for i in human_valid_idx]
human_test_sequences = [human_sequences[i] for i in human_test_idx]

# Truncate longest sequences
human_train_sequences[6640]=human_train_sequences[6640][:5000]
human_train_sequences[6613]=human_train_sequences[6613][:5000]

# Create lengths for sequence representation averaging in FastText
human_train_lengths = dp.sequence_lengths_with_kmers(human_train_sequences,opt.k)
human_valid_lengths = dp.sequence_lengths_with_kmers(human_valid_sequences,opt.k)
human_test_lengths = dp.sequence_lengths_with_kmers(human_test_sequences,opt.k)

#### Yeast sequences

# Load yeast sequences and training data
yeast_sequences, yeast_protein_names = dp.load_FASTA(opt.main_data_dir+'yeast_sequences.fasta')
yeast_train_idx, yeast_valid_idx, yeast_test_idx, yeast_train_labels, yeast_valid_labels,     yeast_test_labels, yeast_GO_terms = dp.load_test_sets(opt.main_data_dir+'yeast_MF_temporal_holdout.mat')

# Create train, validation, and test sets from the full list of yeast proteins
yeast_train_sequences = [yeast_sequences[i] for i in yeast_train_idx]
yeast_valid_sequences = [yeast_sequences[i] for i in yeast_valid_idx]
yeast_test_sequences = [yeast_sequences[i] for i in yeast_test_idx]

# Create lengths for sequence representation averaging in FastText
yeast_train_lengths = dp.sequence_lengths_with_kmers(yeast_train_sequences,opt.k)
yeast_valid_lengths = dp.sequence_lengths_with_kmers(yeast_valid_sequences,opt.k)
yeast_test_lengths = dp.sequence_lengths_with_kmers(yeast_test_sequences,opt.k)

#### Combine sequences

# SEQUENCES
human_train_sequences.extend(yeast_train_sequences)
human_valid_sequences.extend(yeast_valid_sequences)
human_test_sequences.extend(yeast_test_sequences)

combined_train_sequences = human_train_sequences[:]
combined_valid_sequences = human_valid_sequences[:]
combined_test_sequences = human_test_sequences[:]

# Combined kmers
k_mers_combined = dp.get_k_mers(combined_train_sequences,combined_valid_sequences,combined_test_sequences,opt.k)

# LENGTHS
combined_train_lengths = torch.cat([human_train_lengths,yeast_train_lengths])
combined_valid_lengths = torch.cat([human_valid_lengths,yeast_valid_lengths])
combined_test_lengths = torch.cat([human_test_lengths,yeast_test_lengths])

#LABELS
def combine_labels(human_labels,yeast_labels):
    humdf = pd.DataFrame(human_labels,columns=human_GO_terms)
    ystdf = pd.DataFrame(yeast_labels,columns=yeast_GO_terms)
    combined = pd.concat([humdf,ystdf],ignore_index=True).fillna(0)
    return torch.from_numpy(np.array(combined)).type(torch.LongTensor)

combined_train_labels = combine_labels(human_train_labels,yeast_train_labels)
combined_valid_labels = combine_labels(human_valid_labels,yeast_valid_labels)
combined_test_labels = combine_labels(human_test_labels,yeast_test_labels)

# Convert protein sequence strings into long tensors where each int corresponds
# to one of 22 amino acids
combined_train_tensors = dp.TransformAAsToTensor_with_kmers(combined_train_sequences,opt.k,k_mers_combined,acid_dict_human)
combined_valid_tensors = dp.TransformAAsToTensor_with_kmers(combined_valid_sequences,opt.k,k_mers_combined,acid_dict_human)
combined_test_tensors = dp.TransformAAsToTensor_with_kmers(combined_test_sequences,opt.k,k_mers_combined,acid_dict_human)




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


def train(valid_tensors, valid_labels, valid_lengths,num_epochs, optimizer, data_iter, model, training_length):
    losses = []
    total_batches = int(training_length/ opt.batch_size) #375

    validate_every = int((opt.eval_every/100)*total_batches)
    show_every = int((opt.print_every/100)*total_batches)
    val_outputs = None
    valid_label = None
    eval_loss = None

    validation_losses = []

    epoch=0
    #for epoch in range(1, num_epochs+1):
    while epoch <= num_epochs:    
        stop_training = False
        for i, (train_data, train_labels, length_batch) in enumerate(data_iter):
                                                # train_data size: (26, 34350) ; train_label size: (26, 147)
                                                # This needs to be modified. Max length is batch specific !!!!!
            model.train(True)
            model.zero_grad()
            outputs = model(train_data.transpose(0,1).cuda())
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
                flag = True
                while(flag == True):
                    valid_sequences, valid_label,_ = reduced_set(valid_tensors, valid_lengths, valid_labels, 100)
                    if (valid_sequences.size()[0] < 4000):
                        break
                model.eval()
                val_outputs = model(Variable((valid_sequences).transpose(0,1).type(torch.LongTensor).cuda(), volatile=True))
                eval_loss = criterion(val_outputs.cpu().data, valid_label.float())
                validation_losses.append(eval_loss.data[0])
                stop_training = early_stop(validation_losses, 5)

            # Print statements
            if stop_training:
                print("earily stop triggered")
                break
            if (i+1) % show_every == 0:
                print('Epoch: [{0}/{1}], Step: [{2}/{3}], Train loss: {4}, Validation loss:{5}'.format(
                           epoch, num_epochs, i+1, total_batches, np.mean(losses)/(total_batches*epoch), np.mean(np.array(validation_losses))))
        
            if (i+1) % total_batches == 0:
                epoch+=1
                evaluate_and_save(model,opt.out_dir, val_outputs, valid_label, losses, eval_loss.data[0], "combined", epoch)
                break
        if stop_training == True:
            break



#############################
### 6. Learning the model ###
#############################


### Combined ###

data_size = combined_train_labels.shape[0] 
num_labels = combined_train_labels.shape[1] 
if opt.k == 1:
    vocab_size = len(acid_dict_human) + 1
else:
    vocab_size = max(list(k_mers_combined.values())) + 1

model = RNN_GRU(vocab_size, opt.emb_dim, num_labels, opt.hidden_size, n_layers=opt.n_layers, dropout=opt.dropout_dec_p, is_bidirectional=opt.is_bidirectional)

criterion = nn.MultiLabelSoftMarginLoss()

if opt.USE_CUDA == True:
	model = model.cuda()
	criterion = criterion.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.l2_penalty)

data_iter = batch_iter(opt.batch_size, combined_train_tensors, combined_train_labels, combined_train_lengths)

# Model Training
train(combined_valid_tensors, combined_valid_labels, combined_valid_lengths, opt.n_epochs, optimizer, data_iter, model, data_size)



