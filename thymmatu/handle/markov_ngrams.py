#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Markov Ngrams (in development)
    Copyright (C) October 2022 Francesco Camaglia, LPENS 
'''

import numpy as np
import pandas as pd

from thymmatu.handle.ngrams import data_generator, pmf_data_hist_gen

#############
#  ALIASES  #
#############

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

class markov_class() :
    def __init__(
        self, length,
        markov_matrix=None, n_states=None, uniform=False, seed=None
        ) :

        try :
            self.length = int(length)
        except :
            raise IOError("length is an integer greater than 1.")

        if markov_matrix is None :
            if n_states is not None :
                # FIXME check that int >= 1
                self.n_states = n_states
                self.markov_matrix = self.random_Markov_Matrix( uniform=uniform, seed=seed )
                self.is_uniform = uniform
            else :
                raise IOError('One between markov_matrix and n_states must be specified.')
        else :
            # FIXME: add a check
            assert markov_matrix.shape[0] == markov_matrix.shape[1]
            markov_matrix = normalize_matrix(markov_matrix) #FIXME: raise warning
            self.markov_matrix = markov_matrix
            self.n_states = len(markov_matrix)
            self.is_uniform = np.all(markov_matrix == np.mean(markov_matrix))

        # note: n. of blank repetitions `blank` should be the decorrelation length 
        # but since dec.length ~ 1./|log( e_val_1 )| and |e_val_1| < 1.
        # dec. length should be o(1) 
        self.blank = 1 # FIXME: is this correct?

        # is this variable _pmf a good strategy ?
        self._pmf = None

    def random_Markov_Matrix( self, uniform=False, seed=None ) :
        '''Random (left) transition matrix for a system with `n` states.'''
        rng = np.random.default_rng( seed )

        n_states = self.n_states
        if uniform is True :
            W = np.ones( ( n_states, n_states ) )
        else :       
            W = rng.random( ( n_states, n_states ) )
        W = normalize_matrix( W )
        return W

    def statState( self, ) :
        '''Stationary state of to the transition matrix `MarkovMatrix`.'''
        
        # FIXME : is it always correct?
        e_val, e_vec = np.linalg.eig( self.markov_matrix )
        # note: eigenvector are by column
        sstate = e_vec[ :, np.isclose(e_val, 1.) ]
        sstate = np.real( sstate / np.sum(sstate) )
        
        return sstate

    def pmf( self, ) :
        '''The probability mass function of each state.'''

        L = self.length
        A = self.n_states
        mm = self.markov_matrix.T.ravel()
        ss = self.statState().ravel()

        prob_cols = np.zeros( (A**L, L) )
        prob_cols[:, 0] = np.repeat( ss, A**(L-1) )
        for i in np.arange(1, L) :
            prob_cols[:, i] = np.tile( np.repeat( mm, A**(L-1-i) ), A**(i-1) )
        pmf = prob_cols.prod( axis=1 )

        return pd.Series(pmf, index=1+np.arange(len(pmf)))
    
    def generate_counts( self, size, seed=None ) :
        '''Generate histogram of `size` counts from the Markov chain itslef.'''

        # FIXME : 
        # (Old) Generate histogram of `size` counts from the Markov chain itself.
        # output = counts_generator( Markov_count_hist_gen_, size, self, seed=seed, thres=thres )
        output = data_generator( data_hist_gen, size, self.pmf() )

        return output

    def exact_shannon( self, ) :
        '''exact Shannon entropy with stationary state as initial.'''

        L = self.length
        sstate = self.statState( )
        mmatrix = self.markov_matrix

        exact = entr_operator( sstate )[0]
        if L > 1 :
            exact += (L - 1) * entr_operator( mmatrix ).dot( sstate )[0]

        return exact

    def exact_simpson( self, ) :
        '''exact Simpson index with stationary state as initial.'''

        L = self.length
        sstate2 = np.power(self.statState( ), 2)
        mmatrix2 = np.power(self.markov_matrix, 2)

        exact = sstate2
        for _ in range(1, L, 1) :
            exact = mmatrix2.dot( exact )

        return np.sum( exact )

    def exact_kullbackleibler( self, markov_obj2 ) :
        '''exact Kullback-Leibler divergence with stationary states as initial.'''
        return _exact_kullbackleibler( self, markov_obj2 )

# >>>>>>>>>>>>>>>>>>>
#  Other Functions  #
# <<<<<<<<<<<<<<<<<<<

def data_hist_gen( seed, size, *chg_args ) :
    '''Alias waiting for FIXME'''
    pmf = chg_args[0]
    return pmf_data_hist_gen( pmf, size=size, seed=seed )

def Markov_count_hist_gen_( seed, size, *chg_args ) :
    # FIXME : is this correct? especially `blank`?
    
    rng = np.random.default_rng( seed )
    N = int( size ) 

    markov_obj = chg_args[0]
    MarkovMatrix = markov_obj.markov_matrix
    L = markov_obj.length
    blank = markov_obj.blank

    n_states = MarkovMatrix.shape[0]
    
    #  CUMULANT MATRIX  #
    # used for tower sampling
    C = np.copy( MarkovMatrix )
    for j in range( n_states ):
        for i in range( 1, n_states ):
            C[i, j] = C[i-1, j] + MarkovMatrix[i, j]

    #  SAMPLING  #   
    # FIXME : can simply call random choice on the full ngram proba?
    
    randLUT = rng.random( N * (blank + L) )
    old_state = rng.choice(n_states) # initial state  

    sequences = np.zeros( N )
    for n in range(N) :
        for rep in range(blank + L):
            # TOWER SAMPLING with BISECTION
            indx = int( 0.5 * n_states )
            pp = randLUT[ rep + n * (blank + L) ]
            #out of interval condition
            while np.logical_or(pp >= C[indx, old_state], pp < C[indx-1, old_state] ) and indx > 0:
                indx += 1 - 2 * (pp < C[indx, old_state])
            old_state = indx
            
            if rep >= blank:
                sequences[n] += old_state * np.power( n_states, rep - blank )
        
    tmp = pd.Series( sequences ).astype(int)
    output = tmp.groupby(tmp).size()
    output.index = output.index + 1
    
    return output


def _exact_kullbackleibler( markov_obj1, markov_obj2 ) :
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
