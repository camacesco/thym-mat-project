#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Common system commands.
    Copyright (C) February 2022 Francesco Camaglia, LPENS 
'''

import os
import operator
import functools
import numpy as np

###############
#  FILESCOPE  #
###############

class fileScope:
    '''
    A class to extract all possible file informations from string .
    '''
    def __init__( self, fileName ):

        fileName=str(fileName)
        
        # check wheter compressed
        if fileName.endswith('.gz'): 
            self.compression = 'gzip'
            self.ext = ".gz"
        elif fileName.endswith('.zip'): 
            self.compression ='zip'
            self.ext = ".zip"
        else:
            self.compression = None
            self.ext = ""

        # check non-compressed file extension
        if fileName.endswith( '.tsv' + self.ext ):
            self.ext = '.tsv' + self.ext
            self.delimiter = '\t'
            self.header = 0
        elif fileName.endswith( '.csv' + self.ext ):  
            self.ext = '.csv' + self.ext
            self.delimiter = ','
            self.header = 0
        elif fileName.endswith( '.data' + self.ext ):  
            self.ext = '.data' + self.ext
            self.delimiter = ' '
            self.header = 0
        elif fileName.endswith( '.txt' + self.ext ):
            self.ext = '.txt' + self.ext
            self.delimiter = ';'
            self.header = None
        else :
            raise IOError("Input file extension not recognized: please provide a delimiter.")
###

##############################
#  AVERAGE AND STD DEVIATION  #
##############################

def avg_and_std(values, weights=None, ddof=0):
    '''
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    '''

    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average( np.power(values-average,2), weights=weights)
    assert ddof < len(values)
    variance /= 1 - ddof / len(values)
    return (average, np.sqrt(variance))

#######################
#  NON UNIFORM ROUND  #
#######################

def non_unif_round( values, digits=None, ) :

    decimal_values = values - np.floor(values)

    if np.any(digits) == None :
        mydigits = np.abs(np.floor(np.log10(np.abs(decimal_values)))).astype(np.int8)
    else :
        assert len(values) == len(digits)
        mydigits = digits

    rounded_values = np.array(list(map( lambda x : np.round(x[0], x[1]), zip(values.ravel(), mydigits.ravel()))))
    rounded_values = rounded_values.reshape( values.shape )
    if np.any(digits) == None :
        return rounded_values, mydigits
    else :
        return rounded_values

################
#  REDUCELIST  #
################

def reduceList( ListOfLists ):
    '''
    To transform a list of sub-lists in a single list containing the elements of the non-empty sub-lists.
    e.g. 
    ----
    ListOfLists = [ [ 2, 4 ], [ ], [ "A", "G" ] ]
    returns : [ 2, 4, "A", "G" ]
    '''
    
    types = set(list(map(lambda x : type(x),  ListOfLists ) ) )
    if list not in types :
        # nothing to do in this case
        return ListOfLists
    elif types == {list} :
        return functools.reduce( operator.iconcat, ListOfLists, [] )
    else :
        raise TypeError('Impossible to reduce this kind of list.')
###



################
#  TRYMAKEDIR  #
################

def tryMakeDir( path_choice ):
    # create a directory
    if not os.path.exists( path_choice ):
        try: 
            os.makedirs( path_choice, mode=0o777, exist_ok=True ) 
        except OSError as error: 
            print(error)
    else :
        pass
        #print('The folder "' + path_choice + '" already exists: directory creation ignored.'  )
###
