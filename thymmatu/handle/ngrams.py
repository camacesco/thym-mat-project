#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Ngrams
    Copyright (C) February 2022 Francesco Camaglia, LPENS 
'''

import warnings
import numpy as np
import pandas as pd
import string
import multiprocessing

from thymmatu.utils import fileScope, reduceList
from kamapack.shannon import Experiment

##################################
#  DEFAULT ALPHABET DEFINITIONS  #
##################################

# alphabets dictionary
_Alphabet_ = { 
    'NT': (['A', 'C', 'G', 'T'], "nucleotide"),
    'AA': (['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'], "amminoacid"),
    'AA_Stop': (['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y', '*'], "amminoacid + stopping codon"),
    'ASCII_lower': (list(string.ascii_lowercase), "ASCII lowercase"),
    'ASCII_upper': (list(string.ascii_uppercase), "ASCII uppercase"),
    }

############################
#  NGRAM EXPERIMENT CLASS  #
############################

class ngram_gear:
    
    def __init__( self, num=None, alph=None, ngrams_file_input=None ):
 
        '''
        Parameters
        ----------
                
        num: scalar
                the number of letters which form a ngram. It must be an integer greater than 0.
                Ignored if <ngrams_file_input> is chosen.        
                
        alph: str
                the alphabet where the elements belong to. Ignored if <ngrams_file_input> is chosen. 
                The implemented options are:
                - "AA" : amino acid alphabet (20 letters);
                - "AA_Stop" : amino acid alphabet with stop codon "*" (21 letters);
                - "NT" : nucleotides (4 letters);
                - "ASCII_lower" : lower case ASCII alphabet;
                - "ASCII_upper" : upper case ASCII alphabet;                
                
        ngrams_file_input: path/to/file, option
                load the dictionary with counts has saved in "path/to/file.csv".
                       
        Attributes
        ----------
        num :
        alph :
        categories :
        data_hist :
        experiment :
        '''
        
        # >>>>>>>>>>>>>>
        #  INPUT LOAD  #
        # >>>>>>>>>>>>>>

        if ngrams_file_input is not None: 
            #  load parameters from file 
            self.num, self.alph, self.data_hist = load_file_dict( ngrams_file_input ) 

        else :
            # load parameters from user
            
            #  num  #
            if num is None :    
                raise IOError('num must specified if ngrams_file_input is not.')
            elif type( num ) != int : 
                raise TypeError('num must be an integer.')
            elif num < 1 : 
                raise IOError('num must be greater than 0.')
            else : 
                self.num = num
            
            #  alph  #
            if alph is None :    
                raise IOError('alph must specified if ngrams_file_input is not.')
            elif type( alph ) != str : 
                raise TypeError('alph must be a string.')
            elif alph not in list( _Alphabet_.keys( ) ) :
                raise IOError('Alphabet unknown. Options are : '+str(list(_Alphabet_.keys())) )
            else : 
                self.alph = alph 
                
            # assign empty data_hist
            self.data_hist = pd.Series()
 
        self.categories = np.power( len( _Alphabet_[ self.alph ][0] ), self.num )               
        self.experiment = Experiment( self.data_hist, categories=self.categories )

    '''
    Methods
    -------
    '''
    
    # >>>>>>>>>>>>>>>
    #  ASSIGN HICT  #
    # >>>>>>>>>>>>>>>
    
    def assign_hist( self, data_hist ) :
        '''
        Assign the attribute <data_hist> which must be a pandas Serie
        with ngram for index and count in values.
        '''
        
        # WARNING!: missing a check for the user data_hist alphabet
        
        self.data_hist = data_hist          
        self.experiment = Experiment( data_hist, categories=self.categories )
    ###
  
    # >>>>>>>>>>>>>>>
    #  HIST UPDATE  #
    # >>>>>>>>>>>>>>>

    def hist_update( self, sequences, file_output=None, skip=None, beg=None, end=None ):
        '''
        It updates data hist computing ngrams on each entry of the list "sequences".
        Be careful : there is no control on sequences 

        Parameters
        ----------
        sequences: list
                the list of sequences from which ngrams are extracted.
                
        skip: scalar:
                the number of letters to skip after each ngram before considering the next one.
                If skip is set to <num>-1, ngrams are considered one after the other (from the left).
                
                Default is 0. 
        beg: scalar:
                constant number of letters to skip at the beginning of each sequence. Default is 0. 
                
        end: scalar:
                constant number of letters to skip at the end of each sequence. Default is 0. 
                
        file_output: path/to/file.csv.gz, optional
                the path/to/file.csv.gz where to save the output (otherways no file is produced). 
        '''                

        # >>>>>>>>>>>>>>
        #  INPUT LOAD  #
        # >>>>>>>>>>>>>>
        
        #  skip  #
        if skip is None : skip = 0 # Default
        elif type( skip ) != int : raise TypeError('"skip" must be an integer.')
        elif skip < 0 : raise IOError('"skip" must be greater or equal to 0.')
        else : pass 

        #  beg  #
        if beg is None : beg = 0 # Default
        elif type( beg ) != int : raise IOError('"beg" must be an integer.')
        elif beg < 0 : raise IOError('"beg" must be greater or equal to 0.')
        else : pass

        #  end  #
        if end is None : end = 0 # Default
        elif type( end ) != int : raise IOError('"end" must be an integer.')
        elif end < 0 : raise IOError('"end" must be greater or equal to 0.')
        else :  pass
                
        # >>>>>>>>>>>>>
        #  EXECUTION  #
        # >>>>>>>>>>>>> 
                        
        #  UPDATE DICTIONARY from SEQUENCES 
        # WARNING!: maybe parallel here is useless
        POOL = multiprocessing.Pool( multiprocessing.cpu_count() ) 
        results = POOL.starmap( inSequence, [ ( Seq, self.num, skip, beg, end ) for Seq in sequences ] )
        POOL.close()
        
        list_of_ngrams = pd.Series(reduceList(results))
        update_hist = list_of_ngrams.groupby(list_of_ngrams).size()
        
        if not update_hist.empty :         
            if not self.data_hist.empty : 
                self.data_hist = update_hist.add(self.data_hist, fill_value=0)
            else :
                self.data_hist = update_hist                          
        else : 
            warnings.warn("No ngrams returned from the sequences.")  
            
        self.experiment = Experiment( self.data_hist, categories=self.categories )

        #  SAVING FILEOUT 
        if file_output : 
            self.save_file_dict( file_output )
    
    ###
    
    # >>>>>>>>>>>>>>>>>>>>>>>>
    #  SAVE FILE DICTIONARY  #
    # >>>>>>>>>>>>>>>>>>>>>>>>

    def save_file_dict( self, file_output ) :
        '''
        Save n-grams dictionary to a gzipped file.
        '''

        if type( file_output ) is str : 
            if len(file_output.split(".")) > 0 :
                file_output = file_output.split(".")[0] + ".csv.gz"
            else :
                file_output = file_output + ".csv.gz"
        else : 
            raise IOError( 'Unrecognized filename : ' + file_output )

        self.data_hist.to_csv( file_output, header=False, index=True, sep=",", compression="gzip" )        
    ###
    
###


################
#  INSEQUENCE  #
################

def inSequence( thisSeq, num, skip, beg, end ):
    '''
    It returns the list of <num>-grams contained in thisSeq[beg, len(thisSeq)-end].
    Distance between grams is set by the <skip> parameter.
    It returns an empty list if no ngram can be extracted.
    
    e.g.
    ----    
        thisSeq :  A B C D E F G H
        indeces :  0 1 2 3 4 5 6 7

        num=3, skip=1, beg=1, end=2
        returns: [ "BCD", "DEF" ]

        num=4, skip=0, beg=3, end=1
        returns: [ "DEFG" ]

        num=4, skip=0, beg=3, end=2
        returns: [ ]
    '''

    first_indx = beg
    last_indx = len( thisSeq ) - end - num
    
    return [ thisSeq[ i : num + i ] for i in range ( first_indx , 1 + last_indx , 1 + skip ) ]  
###



##########################
#  LOAD FILE DICTIONARY  #
##########################

def load_file_dict( file_input ) :
    '''
    To open the ngrams dictionary <file_input>.
    '''
    
    # load dictionary
    in_scope = fileScope( file_input )
    df = pd.read_csv( file_input, header=None, index_col=0, na_filter = False,
                     compression=in_scope.compression, sep=in_scope.delimiter )
    
    # CHECK num :
    Lengths = list( map( len, df.index.astype(str) ) ) 
    if len(Lengths) == 0 :
        raise IOError( "The ngrams file is empty." )
    elif len( set(Lengths) ) > 1 :
        raise IOError( "The ngrams file contains ngrams with multiple lengths." )
    else :
        num = Lengths[0]
    
    # CHECK alph :
    thisAlph = set(reduceList([[c for c in i] for i in df.index]))
    for alph in _Alphabet_ :
        superset = set( _Alphabet_[ alph ][0] )
        if thisAlph.issubset( superset ) : break
        else : alph = None
    if alph is None:
        raise IOError( "The ngrams file alphabet is not available." )
    
    return num, alph, df
###
