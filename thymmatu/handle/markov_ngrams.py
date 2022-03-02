#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Markov Ngrams (in development)
    Copyright (C) February 2022 Francesco Camaglia, LPENS 
'''

import numpy as np
import pandas as pd
import multiprocessing

def statState( MarkovMatrix ) :
    '''Stationary state of to the transition matrix `MarkovMatrix`.'''
    
    e_val, e_vec = np.linalg.eig( MarkovMatrix )
    # note: eigenvector are by column
    sstate = e_vec[ :, np.isclose(e_val, 1.) ]
    sstate = np.real( sstate / np.sum(sstate) )
    
    return sstate

def probLogprob( x, y ) :
    '''- x * log( x )'''
    # FIXME: zeros in log
    
    return - x * np.log(y)

def random_Markov_Matrix( n_states, uniform=False ) :
    '''Random (left) transition matrix for a system with `n` states.'''
    
    if uniform is True :
        A = np.ones(( n_states, n_states ))
    else :       
        A = np.random.rand( n_states, n_states )
        
    # normalization (rem: P(i->j) = A_{ji}
    for i in range( n_states ):
        Norm = np.sum( A[:, i] )
        A[:, i] = A[:, i] / Norm
        
    return A

def produce_Markov_counts( MarkovMatrix, L, N=1e4, blank=1, seed=None ) :
    # note: n. of blank repetitions `BLANK` should be the decorrelation length 
    # but since dec.length ~ 1./|log( e_val_1 )| and |e_val_1| < 1.
    # dec. length should be o(1)    
    # FIXME : seed must be a vector!

    CPU_count = multiprocessing.cpu_count()
    
    size_per_job = np.min( [ N / CPU_count, 1e5 ] )
    number_of_jobs = int( np.floor( N / size_per_job ) )
    n_pool = size_per_job * np.ones( number_of_jobs )
    n_pool = n_pool.astype(int)
    n_pool[-1] = N - np.sum(n_pool[:-1])

    POOL = multiprocessing.Pool(  )
    args = [ (MarkovMatrix, L, n, blank, seed) for n in n_pool ]
    results = POOL.starmap( _produce_Markov_counts_aux_, args )
    POOL.close()    
    
    output = results[0]
    for to_add in results[1:] :
        output = output.add(to_add, fill_value=0)
    return output

#    
def _produce_Markov_counts_aux_( MarkovMatrix, L, N, blank, seed ) :

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
    for n in range(N) :
        for rep in range(blank + L):
            # TOWER SAMPLING with BISECTION
            indx = int( 0.5 * n_states )
            pp = np.random.random()
            #out of interval condition
            while np.logical_or(pp >= C[indx, old_state], pp < C[indx-1, old_state] ) and indx > 0:
                indx += 1 - 2 * (pp < C[indx, old_state])
            old_state = indx
            
            if rep >= blank:
                sequences[n] += old_state * np.power( n_states, rep - blank )
        
    output = pd.Series( sequences ).astype(int)
    
    return output.groupby(output).size()

# 
def entr_operator( x ) :
    '''Sum over the rows of x * log(x).'''
    
    return np.sum( probLogprob(x,x), axis=0)

#
def Markov_Exact_Shannon( MarkovMatrix, L ) :
    '''
    Computation of the Shannon entropy for L-grams generated through a Markov chain
    with transition matrix equal to `MarkovMatrix`.
    '''
    
    assert L > 0
    sstate = statState( MarkovMatrix )

    S_ex = entr_operator( sstate )[0]
    if L > 1 :
        S_ex += (L - 1) * entr_operator( MarkovMatrix ).dot( sstate )[0]

    return S_ex
    

def cross_entr_operator( x, y ) :
    '''Sum over the rows of x * log(y).'''
    
    return np.sum( probLogprob(x,y), axis=0)


def Markov_Exact_Kullback_Leibler( MarkovMatrix_A, MarkovMatrix_B, L ) :
    '''
    Computation of the Kullback_Leibler divergence for L-grams generated through Markov chains
    with transition matrices equal to `MarkovMatrix_A` and `MarkovMatrix_B`.
    '''
    assert L > 0

    if np.all( MarkovMatrix_A == MarkovMatrix_B ) :
        output = 0.
    
    else :
        sstate_A = statState( MarkovMatrix_A )
        sstate_B = statState( MarkovMatrix_B )

        S_ex_A = Markov_Exact_Shannon( MarkovMatrix_A, L )
        H_ex = cross_entr_operator( sstate_A, sstate_B )[0]
        if L > 1 :
            H_ex += (L-1) * cross_entr_operator( MarkovMatrix_A, MarkovMatrix_B ).dot( sstate_A )[0]

        output = H_ex - S_ex_A
        
    return output
    