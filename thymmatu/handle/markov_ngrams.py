#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Markov Ngrams (in development)
    Copyright (C) November 2021 Francesco Camaglia, LPENS 
'''

import numpy as np
import pandas as pd
from numpy.linalg import matrix_power
import tqdm

#
def statState( MarkovMatrix ) :
    '''
    It returns the stationary state according to the transition matrix `MarkovMatrix`.
    '''
    
    e_val, e_vec = np.linalg.eig( MarkovMatrix )
    # note: eigenvector are by column
    sstate = e_vec[ :, np.isclose(e_val, 1.) ]
    sstate = np.real( sstate / np.sum(sstate) )
    
    return sstate

#
def probLogprob( x, y ) :
    '''
    $$ - x * log( x ) $$ 
    '''
    
    # WARNING!: zeros in log
    
    return - x * np.log(y)

# 
def random_Markov_Matrix( n_states, uniform=False ) :
    '''
    Return a random (left) transition matrix for a system with number of states equal to `n` .
    '''
    
    if uniform is True :
        A = np.ones(( n_states, n_states ))
    else :       
        A = np.random.rand( n_states, n_states )
        
    # normalization (rem: P(i->j) = A_{ji}
    for i in range( n_states ):
        Norm = np.sum( A[:, i] )
        A[:, i] = A[:, i] / Norm
        
    return A

#    
def produce_Markov_counts( MarkovMatrix, L, N=1e4, BLANK=1, seed=None ) :
    # note: n. of blank repetitions `BLANK` should be the decorrelation length 
    # but since dec.length ~ 1./|log( e_val_1 )| and |e_val_1| < 1.
    # dec. length should be o(1)
    
    np.random.seed( seed=seed )
    
    n_states = MarkovMatrix.shape[0]
    
    #  CUMULANT MATRIX  #
    # used for tower sampling
    C = np.copy( MarkovMatrix )
    for j in range( n_states ):
        for i in range( 1, n_states ):
            C[i, j] = C[i-1, j] + MarkovMatrix[i, j]

    #  SAMPLING  #
    old_state = int( np.random.random() * n_states ) # initial state           
            
    N = int(N)
    sequences = np.zeros( N )
    for n in tqdm.tqdm(range(N), desc='Sequences generation') :
        for rep in range(BLANK + L):
            # TOWER SAMPLING with BISECTION
            indx = int( 0.5 * n_states )
            pp = np.random.random()
            #out of interval condition
            while np.logical_or(pp >= C[indx, old_state], pp < C[indx-1, old_state] ) and indx > 0:
                indx += 1 - 2 * (pp < C[indx, old_state])
            old_state = indx
            
            if rep >= BLANK:
                sequences[n] += old_state * np.power( n_states, rep - BLANK )
        
    output = pd.Series( sequences ).astype(int)
    
    return output.groupby(output).size()

# 
def entr_operator( x ) :
    '''
    Sum over the rows of x * log(x)
    '''
    
    return np.sum( probLogprob(x,x), axis=0)

#
def Markov_Exact_Shannon( MarkovMatrix, L ) :
    '''
    Computation of the Shannon entropy for L-grams generated through a Markov chain
    of a Markov transition matrix equal to `MarkovMatrix`.
    '''
    
    assert L > 0
    sstate = statState( MarkovMatrix )

    S_ex = entr_operator( sstate )[0]

    if L > 1 :
        i = 1
        while (i < L) :
            temp = matrix_power( MarkovMatrix, L-2 ).dot( sstate )
            S_ex += entr_operator( MarkovMatrix ).dot( temp )[0]
            i += 1

    return S_ex
    
# 
def cross_entr_operator( x, y ) :
    '''
    Sum over the rows of x * log(y)
    '''
    
    return np.sum( probLogprob(x,y), axis=0)

#
def Markov_Exact_Kullback_Leibler( MarkovMatrix_A, MarkovMatrix_B, L ) :
    '''
    '''
    if np.all( MarkovMatrix_A == MarkovMatrix_B ) :
        output = 0.
    
    else :
        S_ex_A = Markov_Exact_Shannon( MarkovMatrix_A, L )

        assert L > 0
        sstate_A = statState( MarkovMatrix_A )
        sstate_B = statState( MarkovMatrix_B )

        H_ex = cross_entr_operator( sstate_A, sstate_B )[0]

        if L > 1 :
            i = 1
            while (i < L) :
                temp = matrix_power( MarkovMatrix_A, L-2 ).dot( sstate_A )
                H_ex += cross_entr_operator( MarkovMatrix_A, MarkovMatrix_B ).dot( temp )[0]
                i += 1

        output = H_ex - S_ex_A
        
    return output
    