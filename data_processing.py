import numpy as np
import pandas as pd
import scipy.io as sio
from Bio import SeqIO
import collections
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


acid_dict = {'A': 8, 'C': 6, 'D': 11, 'E': 2, 'F': 5,
	     'G': 18, 'H': 12, 'I': 17, 'K': 10, 'L': 3,
	     'M': 1, 'N': 19, 'P': 9, 'Q': 14, 'R': 16,
	     'S': 7, 'T': 15, 'U': 22, 'V': 4, 'W': 20,
	     'X': 21, 'Y': 13}

def vectorize_AAs(string):
    '''This function takes an amino-acid string as input and outputs a vector of integers, with each
    integer representing one amino acid.
    
    For example, 'BACEA' is converted to [2, 1, 3, 5, 1]
    '''
    character_list = list(string) #converts 'BACEA' to ['B','A','C','E','A]
    for i in range(len(character_list)):
        character_list[i] = acid_dict[character_list[i]] #convert the character to a number
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

def batch_iter(batch_size, sequences, labels, lengths=None):
    start = -1 * batch_size
    dataset_size = sequences.size()[0]
    order = list(range(dataset_size))
    random.shuffle(order)

    while True:
        start += batch_size
        if start > dataset_size - batch_size:
            # Start another epoch.
            start = 0
            random.shuffle(order)
        batch_indices = order[start:start + batch_size]
        batch_indices_tensor = torch.LongTensor(batch_indices)
        batch_train = sequences[batch_indices_tensor].type(torch.LongTensor)
        batch_train_labels = labels[batch_indices_tensor]
        if lengths is not None:
            length_batch = lengths[batch_indices_tensor]
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