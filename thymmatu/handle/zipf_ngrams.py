#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Zipf Ngrams (in development)
    Copyright (C) March 2022 Francesco Camaglia, LPENS 
'''

import numpy as np
import pandas as pd
import multiprocessing
from scipy.special import zeta
from kamapack._aux_definitions import get_from_implicit
from tqdm import tqdm

# FIXME : jupyter bug multiprocessing auxiliary definition
from kamapack.default_divergence import Dirichlet
def func_Dirichlet( a, b, compACT ) : 
    return Dirichlet(compACT, a=a, b=b)
# FIXME :

class zipf_class() :

    def __init__(self, gamma, rank=None, n_min=1, n_max=None) :
        '''
        Zipf's law n^{-gamma} for n in [n_min, n_max)
        min included, max excluded
        '''

        # input check #
        assert n_min >= 1
        assert gamma > 1
        if n_max is not None :
            try : n_max = int(n_max)
            except : raise TypeError("n_max") # FIXME

        if rank is None :
            if n_max is None :
                pass
            else :
                rank = np.arange(n_min, n_max)
        else :
            n_min = np.min(rank)
            n_max = np.max(rank) + 1

        # input assignment #
        self.gamma = gamma
        self.rank = rank
        self.n_min = n_min
        self.n_max = n_max

    def pmf( self, n, ) :
        '''Zipf's law probability mass function.'''

        gamma = self.gamma
        norm = zeta(gamma, self.n_min) 
        if self.n_max is not None :
            norm -= zeta(gamma, self.n_max)

        return 1. / ( np.power(n, gamma) * norm )

    def exact_shannon( self, ) :
        '''Exact entropy for a Zipf's law (natural log).'''

        n = np.arange(self.n_min, self.n_max, 1)
        p_n = self.pmf( n, )
        return (-1) * np.sum( p_n * np.log( p_n ) )

    def generate_counts( self, N ) :
        '''Zipf generated counts.
        N : number of counts
        gamma : Zipf's exponent
        rank (optional) : the rank-ordered category names'''

        try : N = int(N)
        except : raise TypeError("N is a postive integer.") # FIXME

        serie = pd.Series( _zipf_rand_gen( self, N ) )
        serie = serie.groupby(serie).size()
        serie.index = [self.rank[i-1] for i in serie.index.values]

        return serie


def _zipf_rand_gen( zipf_obj, size, verbose=False ):
    '''Zipf's law approximate random number generator.'''
    disable = not verbose

    gamma, n_min, n_max = zipf_obj.gamma, zipf_obj.n_min, zipf_obj.n_max

    t_1 = zeta(gamma, n_min)
    t_2 = zeta(gamma, n_max)
    rnd_vec = np.random.rand(size)
    args = [ (_implict_cdf_zipf, r, n_min, n_max, gamma, t_1, t_2) for r in rnd_vec ]
    POOL = multiprocessing.Pool( multiprocessing.cpu_count() )  
    params = POOL.starmap( get_from_implicit, tqdm(args, total=len(args), desc='Generation', disable=disable) )
    POOL.close()
    return np.floor(np.asarray( params )).astype(int)

def _implict_cdf_zipf(x, r, gamma, t_1, t_2 ) :
    '''auxiliary: implicit relation to be inverted for transform method.'''
    norm = t_1 - t_2
    return (t_1 - zeta(gamma, x)) - r * norm



def exact_kullbackleibler(zipf_obj1, zipf_obj2, ) :
    '''Exact Kullback-Leibler divergence for two Zipf's laws.'''
    # exec

    assert zipf_obj1.n_min == zipf_obj2.n_min 
    assert zipf_obj1.n_max == zipf_obj2.n_max 

    index = np.arange(zipf_obj1.n_min, zipf_obj1.n_max)

    p_1 = zipf_obj1.pmf( index )
    p_2 = zipf_obj2.pmf( index )

    serie_1 = pd.Series(p_1, index=zipf_obj1.rank,)
    serie_2 = pd.Series(p_2, index=zipf_obj2.rank,)
    df = pd.concat( [serie_1, serie_2], axis=1 )

    p_1 = df[0].values
    p_2 = df[1].values

    exact = np.sum(p_1 * np.log(p_1/p_2))

    return exact