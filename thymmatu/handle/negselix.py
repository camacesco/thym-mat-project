#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import Levenshtein # run : pip install levenshtein

#  Amino Acid alphabet encoder  #
aa_alphabet = list('ACDEFGHIKLMNPQRSTVWY')
AA_ENCODE = dict(zip(aa_alphabet, range(len(aa_alphabet))))

#  Global Variables  #
STOCHASTIC_APOPTOSIS = 5.e-8

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# class for the selection model #
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

class selection_model( ) :
    '''A class to recollect all the definitions of the selection model.'''
    def __init__( 
            self, 
            encouter_proba,
            distance_scale,
            stochastic_apoptosis,
        ) :
        self.k = encouter_proba
        self.d0 = distance_scale
        self.a = stochastic_apoptosis

        # auxiliary constant
        self._norm = encouter_proba / (1. - stochastic_apoptosis )

    def single_death_proba( self, distance ) :
        '''Probability of commiting apoptosis for a given distance.'''
        return self.k * np.exp( - distance / self.d0 ) + self.a
    
    def _norm_single_surv_proba( self, distance ) :
        '''The normalized single survival probability. Meant for simulation purposes to be multuplied to p_star.'''
        return 1. - self._norm * np.exp( - distance / self.d0 )
    
    def hamming( self, seq_1, seq_2, norm=False ) :
        '''Hamming distance between two encoded sequences. Default is not normalized.'''
        # FIXME : for the moment just one of the two can be a list of sequences
        # FIXME : control on array type and lengths
        output = ( np.abs(seq_2 - seq_1) > 0 ).sum( axis=1 )
        if norm is True :
            output = output / len(seq_1)
        return output
        
    def levenshtein( self, seq_1, seq_2 ) :
        '''Levenshtein distance between two encoded sequences.'''
        # FIXME : for the moment just one of the two can be a list of sequences
        is_a_list_1 = (len(np.shape(seq_1)) > 1) or (seq_1.dtype == np.dtype("O"))
        is_a_list_2 = (len(np.shape(seq_2)) > 1) or (seq_2.dtype == np.dtype("O"))
        if is_a_list_1 and is_a_list_2 :
            raise SystemError("Not implemented for returning matrix of distances.")
        elif is_a_list_1 :
            output = np.array(list(map(lambda x : Levenshtein.distance(x, seq_2), seq_1)))
        elif is_a_list_2 :
            output = np.array(list(map(lambda x : Levenshtein.distance(seq_1, x), seq_2)))
        else :
            output = Levenshtein.distance(seq_1, seq_2)
        return output 
    
    def tcrdist( self, seq_1, seq_2 ) :
        '''TCRdist distance between two encoded sequences.'''
        # FIXME : for the moment just one of the two can be a list of sequences
        is_a_list_1 = (len(np.shape(seq_1)) > 1) or (seq_1.dtype == np.dtype("O"))
        is_a_list_2 = (len(np.shape(seq_2)) > 1) or (seq_2.dtype == np.dtype("O"))
        if is_a_list_1 and is_a_list_2 :
            raise SystemError("Not implemented for returning matrix of distances.")
        elif is_a_list_1 :
            output = np.array(list(map(lambda x : Levenshtein.distance(x, seq_2), seq_1)))
        elif is_a_list_2 :
            output = np.array(list(map(lambda x : Levenshtein.distance(seq_1, x), seq_2)))
        else :
            output = Levenshtein.distance(seq_1, seq_2)
        return output 

#############################
#  LOAD DATA + PRE-PROCESS  #
#############################
    
def encode_aa( seq_cdr3aa ) :
    '''It translates amino acid sequences into numbers.'''
    # FIXME : this might just be a slow-down...
    return np.array([AA_ENCODE[c] for c in seq_cdr3aa ])

def class_groupby( classes ) :
    '''Returns a dictionary with the index belonging to a given class.'''
    aux = pd.Series( classes )
    return aux.groupby(aux).groups

def assign_tcr_class( seq, include_len=False ) :
    '''Returns a string for the V,D class of the TCR chain. It can also include the length.'''
    V_gene = seq['V'][3:].replace("/", "_")
    J_gene = seq['J'][3:].replace("/", "_")
    if include_len is True :
        Length = len(seq['aa'])
        output = f"{V_gene}+{J_gene}+L{Length}"
    else :
        output = f"{V_gene}+{J_gene}"
    return output

def open_olga( filename, drop_dupl=None ) :
    '''Opens the file with the generated sequences from Olga.'''
    df = pd.read_csv( filename, header=None, names=["aa", "V", "J", "nt"] )
    df = df.dropna()
    # drop all nucletoides or amino acid duplicates
    if drop_dupl != None :
        if "nt" == drop_dupl :
            df = df.drop_duplicates()
            df = df.drop(columns=["nt"])
        elif "aa" == drop_dupl :
            df = df.drop(columns=["nt"])
            df = df.drop_duplicates()
        else :
            raise IOError("Forbidden drop in `drop_dupl`.")
    df.reset_index(inplace=True, drop=True)
    return df

def preprocess_tcrs( olga, distance='Levenshtein' ) :
    '''Preprocess the opend olga file according to the chosen distance.'''
    output = {}
    output['encoded'] = olga["aa"].apply(encode_aa)
    output['lenghts'] = olga["aa"].apply(len)
    output['size'] = len(olga)
    if distance == 'Hamming' :
        # define classes for Hamming (with length)
        auto_reac_classes_Ham = olga.apply(assign_tcr_class, axis=1, args=(True,))
        output['classes'] = auto_reac_classes_Ham
        output['class_groupby'] = class_groupby(auto_reac_classes_Ham)
    elif distance == 'Levenshtein' :
        # define classes for Levenshtein (no length)
        auto_reac_classes_Lev = olga.apply(assign_tcr_class, axis=1, args=(False,))
        output['classes'] = auto_reac_classes_Lev
        output['class_groupby'] = class_groupby( auto_reac_classes_Lev )
    else :
        raise IOError(f"Unknown distance : `{distance}`. Please choose between `Hamming` and `Levenshtein`.")
    return output

#################################
#  INITIALIZE MODEL PARAMETERS  #
#################################


def simulator( model, autoreac_tcr, test_tcr, distance='Levenshtein' ) :
    '''Given an object `selection_model` '''

    ###############
    #  EXECUTION  #
    ###############

    p_surv = [ ]
    tcr_idx = [ ]
    distances = [ ]

    # loop over the classes found in the test set
    p_star = np.exp(-model.a * autoreac_tcr['size'])
    for tcr_class in test_tcr['class_groupby'] :
        # load auto-reactive of the corresponding class
        auto_reac_idxs = autoreac_tcr['class_groupby'].get(tcr_class, np.array([]))
        auto_reac_cdr3aa_enc = autoreac_tcr['encoded'][auto_reac_idxs].values
        auto_reac_class_size = len(auto_reac_idxs)
        # loop over the sequences in the class
        for seq_cdr3aa_idx in test_tcr['class_groupby'][tcr_class] :
            if auto_reac_class_size == 0 :
                all_dists = np.array([])
                this_p = p_star
            else : 
                if distance == 'Hamming' :             
                    all_dists = model.hamming( test_tcr['encoded'][seq_cdr3aa_idx], np.stack(auto_reac_cdr3aa_enc) )  
                elif distance == 'Levenshtein' :
                    all_dists = model.levenshtein( test_tcr['encoded'][seq_cdr3aa_idx], auto_reac_cdr3aa_enc )   
                else :
                    raise IOError(f"Unknown distance : `{distance}`. Please choose between `Hamming` and `Levenshtein`.")
                this_p = p_star * np.prod( model._norm_single_surv_proba( all_dists ) )
            # store these results
            tcr_idx.append( seq_cdr3aa_idx )
            p_surv.append( this_p )
            distances.append( all_dists )

    output = {}
    output['p_surv'] = np.array([x for _, x in sorted(zip(tcr_idx, p_surv))])
    output['distances'] = [x for _, x in sorted(zip(tcr_idx, distances))]

    return output