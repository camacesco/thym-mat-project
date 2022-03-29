#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Markov Ngrams (in development)
    Copyright (C) February 2022 Francesco Camaglia, LPENS 
'''

import numpy as np
import pandas as pd
import multiprocessing

class markov_class() :
    def __init__(self, length, markov_matrix=None, n_states=None, uniform=False) :

        try :
            self.length = int(length)
        except :
            raise IOError("length is an integer greater than 1.")

        if markov_matrix is None :
            if n_states is not None :
                # FIXME check that int >= 1
                self.n_states = n_states
                self.markov_matrix = self.random_Markov_Matrix( uniform=uniform )
                self.is_uniform = uniform
            else :
                raise IOError('One between markov_matrix and n_states must be specified.')
        else :
            # FIXME: add a check
            assert markov_matrix.shape[0] == markov_matrix.shape[1]
            self.markov_matrix = markov_matrix
            self.n_states = len(markov_matrix)
            self.is_uniform = np.all(markov_matrix == np.mean(markov_matrix))

    def random_Markov_Matrix( self, uniform=False ) :
        '''Random (left) transition matrix for a system with `n` states.'''
        
        n_states = self.n_states

        if uniform is True :
            A = np.ones(( n_states, n_states ))
        else :       
            A = np.random.rand( n_states, n_states )
            
        A = normalize_matrix( A )
            
        return A


    def statState( self, ) :
        '''Stationary state of to the transition matrix `MarkovMatrix`.'''
        
        # FIXME : is it always correct?
        e_val, e_vec = np.linalg.eig( self.markov_matrix )
        # note: eigenvector are by column
        sstate = e_vec[ :, np.isclose(e_val, 1.) ]
        sstate = np.real( sstate / np.sum(sstate) )
        
        return sstate

    def generate_counts( self, N, blank=1, seed=None ) :
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
        args = [ (self.markov_matrix, self.length, n, blank, seed) for n in n_pool ]
        results = POOL.starmap( _produce_Markov_counts_aux_, args )
        POOL.close()    
        
        output = results[0]
        for to_add in results[1:] :
            output = output.add(to_add, fill_value=0)
        return output

    def exact_shannon( self, ) :
        '''
        Computation of the Shannon entropy for L-grams generated through a Markov chain
        with transition matrix equal to `MarkovMatrix`.
        '''

        L = self.length
        sstate = self.statState(  )

        S_ex = entr_operator( sstate )[0]
        if L > 1 :
            S_ex += (L - 1) * entr_operator( self.markov_matrix ).dot( sstate )[0]

        return S_ex

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


def probLogprob( x, y ) :
    '''- x * log( x )'''
    # FIXME: zeros in log
    return - x * np.log(y)

def entr_operator( x ) :
    '''Sum over the rows of x * log(x).'''
    return np.sum( probLogprob(x,x), axis=0)


def cross_entr_operator( x, y ) :
    '''Sum over the rows of x * log(y).'''
    return np.sum( probLogprob(x,y), axis=0)


def exact_kullbackleibler( markov_obj1, markov_obj2 ) :
    '''
    Computation of the Kullback_Leibler divergence for L-grams generated through Markov chains
    with transition matrices equal to `MarkovMatrix_A` and `MarkovMatrix_B`.
    '''

    assert markov_obj1.n_states == markov_obj2.n_states
    assert markov_obj1.length == markov_obj2.length
    L = markov_obj1.length

    if np.all( markov_obj1.markov_matrix == markov_obj2.markov_matrix ) :
        output = 0.
    
    else :
        sstate_1 = markov_obj1.statState( )
        sstate_2 = markov_obj2.statState( )

        entropy_ex_1 = markov_obj1.exact_shannon( )
        crossentropy_ex = cross_entr_operator( sstate_1, sstate_2 )[0]
        if L > 1 :
            crossentropy_ex += (L-1) * cross_entr_operator( markov_obj1.markov_matrix, markov_obj2.markov_matrix ).dot( sstate_1 )[0]

        output = crossentropy_ex - entropy_ex_1
        
    return output
    
def normalize_matrix( A ) :
    '''normalization (rem: P(i->j) = A_{ji}'''
    n_states = A.shape[0]
    for i in range( n_states ):
        Norm = np.sum( A[:, i] )
        A[:, i] = A[:, i] / Norm
    return A