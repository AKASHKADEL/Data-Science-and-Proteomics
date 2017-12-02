import numpy as np
import scipy.io as sio
from Bio import SeqIO
from sklearn import metrics
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import random

def load_test_sets(filename):
    go_data = sio.loadmat(filename, squeeze_me=True)
    go_terms = go_data['goTerm_labels'] # names of gene ontology function terms
    train_annotations = np.asarray(go_data['trainProts_label'].todense()) # training set of function annotations
    valid_annotations = np.asarray(go_data['validProts_label'].todense()) # valid "" ""
    test_annotations = np.asarray(go_data['testProts_label'].todense()) # test "" ""
    train_inds = go_data['trainProts']
    train_inds = train_inds - 1
    valid_inds = go_data['validProts']
    valid_inds = valid_inds - 1
    test_inds = go_data['testProts']
    test_inds = test_inds - 1 # subtract 1 for matlab index conversion into python

    return train_inds, valid_inds, test_inds, train_annotations, valid_annotations, test_annotations, go_terms

def load_FASTA(filename):
    """ Loads fasta file and returns a list of the Bio SeqIO records """
    infile = open(filename)
    full_entries = list(SeqIO.parse(infile, 'fasta'))
    sequences = [str(entry.seq) for entry in full_entries]
    names = [str(entry.id) for entry in full_entries]

    return sequences, names

def sequence_lengths(sequence):
    # From a list of sequences, it returns a tensor length 
    # of every sequence for FastText calculation.

    return torch.LongTensor([len(sequence[i]) for i in range(len(sequence))])

def sequence_lengths_with_kmers(sequence, k):
    # From a list of sequences, it returns a tensor length 
    # of every sequence for FastText calculation. (along with its k_mers)
    list_length = []
    for i in range(len(sequence)):
        if k == 1:
            list_length.append(len(sequence[i]))
        else:
            kmer_len = 0
            for i in range(2, k+1):
                kmer_len += len(sequence[i]) - (i-1)
            val = len(sequence[i]) + kmer_len
            list_length.append(val)

    return torch.LongTensor(list_length)


acid_dict_human = {'A': 8, 'C': 6, 'D': 11, 'E': 2, 'F': 5,
	     'G': 18, 'H': 12, 'I': 17, 'K': 10, 'L': 3,
	     'M': 1, 'N': 19, 'P': 9, 'Q': 14, 'R': 16,
	     'S': 7, 'T': 15, 'U': 22, 'V': 4, 'W': 20,
	     'X': 21, 'Y': 13}

acid_dict_yeast = {'A': 8, 'C': 6, 'D': 11, 'E': 2, 'F': 5,
	     'G': 18, 'H': 12, 'I': 17, 'K': 10, 'L': 3,
	     'M': 1, 'N': 19, 'P': 9, 'Q': 14, 'R': 16,
	     'S': 7, 'T': 15, 'V': 4, 'W': 20, 'Y': 13}


def get_k_mers(train_seq, valid_seqs, test_seqs, k, org='human'):
    '''  Function to get the k-mers indices for all the sequences
         ie, for all the sequences in train, validation & test data
    '''
    k_mers = dict()
    if org=='human':
    	ind = 23 # This is the starting index for k-mers, the individual amino acids terms take the first 26 indices
    else:
    	ind = 21

    total_sequences = train_seq + valid_seqs + test_seqs
    for i in range(len(total_sequences)):
        seq = total_sequences[i]
        for k_val in range(2, k+1):
	        for i in range(len(seq)-(k_val-1)):
	            key = seq[i:i+k_val]
	            if key not in list(k_mers.keys()):
	                k_mers.update({key : ind})
	                ind += 1
                
    return k_mers

def vectorize_AAs(string, dict_amino):
    '''This function takes an amino-acid string as input and outputs a vector of integers, with each
    integer representing one amino acid.
    
    For example, 'BACEA' is converted to [2, 1, 3, 5, 1]
    '''
    character_list = list(string) #converts 'BACEA' to ['B','A','C','E','A]
    for i in range(len(character_list)):
        character_list[i] = dict_amino[character_list[i]] #convert the character to a number
    return character_list

def vectorize_AAs_with_kmers(string, k, k_mers, dict_amino):
    '''This function takes an amino-acid string as input and outputs a vector of integers, with each
    integer representing one amino acid along with its k-mers.
    
    For example, 'BACEA' is converted to [2, 1, 3, 5, 1]
    '''
    character_list = list(string) #converts 'BACEA' to ['B','A','C','E','A]
    for i in range(len(character_list)):
        character_list[i] = dict_amino[character_list[i]] #convert the character to a number

    if k > 1:
    	for k_val in range(2, k+1):

	        for i in range(len(string) - (k_val-1)):
	            key = string[i:i+k_val]
	            character_list.append(k_mers[key])

    return character_list



def AddZeros(vector, max_length):
    '''This function adds the necessary number of zeros and returns an array'''
    #max_length = length of longest vector
    #oldvector = initial vector for that amino-acid chain (in integers)
    if len(vector) >= max_length:
        vector = vector[:max_length]
    else:
        vector.extend(np.zeros(max_length-len(vector)))
        
    return vector 

def TransformAAsToTensor(ListOfSequences,length=None):
    '''This function takes as input a list of amino acid strings and creates a tensor matrix
    of dimension NxD, where N is the number of strings and D is the length of the longest AA chain
    
    "ListOfSequences" can be training, validation, or test sets
    '''
    #find longest amino-acid sequence
    if length is None:
        max_length = len(max(ListOfSequences, key=len))
    else:
        max_length = length
    Sequences = ListOfSequences.copy() 
    for AA in range(len(Sequences)): #for each amino-acid sequence
        Sequences[AA] = vectorize_AAs(Sequences[AA])
        Sequences[AA] = AddZeros(Sequences[AA], max_length)
    NewTensor = torch.from_numpy(np.array(Sequences)).long()
    return NewTensor

def TransformAAsToTensor_with_kmers(ListOfSequences,k,k_mers,dict_amino,length=None):
    '''This function takes as input a list of amino acid strings and creates a tensor matrix
    of dimension NxD, where N is the number of strings and D is the length of the longest AA chain
    
    "ListOfSequences" can be training, validation, or test sets
    '''
    #find longest amino-acid sequence
    if length is None:
        max_length = len(max(ListOfSequences, key=len))

        if k > 1:
            kmer_len = 0
            for i in range(2, k+1):
                kmer_len += max_length - (i-1)
            max_length = max_length + kmer_len    # <<==== EXTRA LINE ADDED TO ACCOUT FOR LENGTH INCREASE DUE TO ADDITION OF KMERS

    else:
        max_length = length
    Sequences = ListOfSequences.copy() 
    for AA in range(len(Sequences)): #for each amino-acid sequence
        Sequences[AA] = vectorize_AAs_with_kmers(Sequences[AA], k, k_mers, dict_amino)  # <<==== use new function which takes k-mers into consideration
        Sequences[AA] = AddZeros(Sequences[AA], max_length)
    NewTensor = torch.from_numpy(np.array(Sequences)).long()
    return NewTensor

def batch_iter(batch_size, sequences, labels, lengths=None):
    start = -1 * batch_size
    dataset_size = sequences.size()[0]
    order = list(range(dataset_size))
    random.shuffle(order)

    while True:
        start += batch_size
        if start > dataset_size - batch_size:
            # Start another epoch.
            # start = 0
            # random.shuffle(order)
            break
        batch_indices = order[start:start + batch_size]
        batch_indices_tensor = torch.LongTensor(batch_indices)
        batch_train = sequences[batch_indices_tensor].type(torch.LongTensor)
        batch_train_labels = labels[batch_indices_tensor]
        if lengths is not None:
            length_batch = lengths[batch_indices_tensor]
            # The 2 lines below will remove the extra zeros which are present after the max_length in the batch
            max_length_batch = length_batch.max()
            batch_train = batch_train[:,:max_length_batch]
            yield [Variable(batch_train), Variable(batch_train_labels), Variable(length_batch)]  
        else:
            yield [Variable(batch_train), Variable(batch_train_labels)]

def eval_iter(batch_size,sequence_tensors,labels):
    '''Returns list of length batch_size, each entry is a 
    tuple with LongTensors of sequences and labels, respectively'''
    batches = []
    dataset_size = len(sequence_tensors)
    start = -1 * batch_size
    order = list(range(dataset_size))
    random.shuffle(order)

    while start < dataset_size - batch_size:
        start += batch_size
        batch_indices = order[start:start + batch_size]
        batch_sequences = [sequence_tensors[i].long() for i in batch_indices]
        batch_labels = [labels[i].long() for i in batch_indices]
        if len(batch_sequences) == batch_size:
            batches.append((torch.stack(batch_sequences),torch.stack(batch_labels)))
        else:
            continue
    return batches


# The below function is useful when we want to extract a smaller subset of data from a larger subset.
def reduced_set(sequence, sequence_lengths, sequence_y, size):

    order = list(range(sequence.size()[0]))
    random.shuffle(order)
    order = order[:size]
    sequence = torch.from_numpy(sequence.numpy()[order])
    sequence_lengths = torch.from_numpy(sequence_lengths.numpy()[order])
    max_len = sequence_lengths[:size].max()
    return sequence[:size, :max_len], sequence_y[:size], sequence_lengths



def evaluate_and_save(output_file,val_outputs,valid_label,losses,val_loss,organism="organism",epoch="epoch"):
    predictions = val_outputs.cpu().data.numpy()
    actual = valid_label.numpy()
    tloss = np.mean(losses)
    vloss = val_loss
    micAUC = em.calculate_AUC(predictions,actual)[0]
    micAUPR = em.calculate_AUPR(predictions,actual)
    macAUC = em.calculate_macro_AUC(predictions,actual)
    macAUPR = em.calculate_macro_AUPR(predictions,actual)
    F1 = em.calculate_micro_F1(predictions,actual)

    torch.save(model.state_dict(),output_file+'%sEpoch%s|TrainLoss:%0.4f|ValLoss:%0.4f|micAUC:%0.4f|micAUPR:%0.4f|macAUC:%0.4f|macAUPR:%0.4f|F1:%0.4f'\
			% (organism,epoch,tloss,vloss,micAUC,micAUPR,macAUC,macAUPR,F1))