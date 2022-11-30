#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Ngrams
    Copyright (C) October 2022 Francesco Camaglia, LPENS 
'''

import warnings
import numpy as np
import pandas as pd
import string
import multiprocessing
from itertools import product

from thymmatu.utils import fileScope, reduceList
from kamapack.estimate import Experiment
from sklearn.preprocessing import FunctionTransformer

##################################
#  DEFAULT ALPHABET DEFINITIONS  #
##################################

# alphabets dictionary
_Alphabet_ = { 
    'NT': (['A', 'C', 'G', 'T'], "nucleotide"),
    'AA': (['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'], "amminoacid"),
    'AA_Stop': (['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y', ''], "amminoacid + stopping codon"),
    'ASCII_lower': (list(string.ascii_lowercase), "ASCII lowercase"),
    'ASCII_upper': (list(string.ascii_uppercase), "ASCII uppercase"),
    }

############################
#  NGRAM EXPERIMENT CLASS  #
############################

class ngram_gear:
    
    def __init__( self, num=None, alph=None, ngrams_file_input=None, skip=None, beg=None, end=None ):
 
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

        skip: scalar
                the number of letters to skip after each ngram before considering the next one.
                If skip is set to <num>-1, ngrams are considered one after the other (from the left).
                Default is 0. 
        beg: scalar
                constant number of letters to skip at the beginning of each sequence. Default is 0. 
                
        end: scalar
                constant number of letters to skip at the end of each sequence. Default is 0. 
                       
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
                raise IOError('`alph` must specified if `ngrams_file_input` is not.')
            elif type( alph ) != str : 
                try :
                    self.alphabet = list(alph)
                except :
                    raise TypeError('`alph` must be a string.')
            elif alph not in list( _Alphabet_.keys( ) ) :
                raise IOError(f"Alphabet unknown. Options are : {list(_Alphabet_.keys())}" )
            else : 
                self.alphabet = _Alphabet_[ self.alph ][0] 
                
            # assign empty data_hist
            self.data_hist = pd.Series()
 
        self.assign_features( skip=skip, beg=beg, end=end )

        self.categories = np.power( len(self.alphabet), self.num )             
        self.experiment = Experiment( self.data_hist, categories=self.categories )

    '''
    Methods
    -------
    '''
    
    # >>>>>>>>>>>>>>>
    #  ASSIGN HIST  #
    # >>>>>>>>>>>>>>>
    
    def assign_hist( self, data_hist ) :
        '''
        Assign the attribute <data_hist> which must be a pandas Serie
        with ngram for index and count in values. FIXME
        '''
        
        # WARNING!: missing a check for the user data_hist alphabet
        
        self.data_hist = data_hist          
        self.experiment = Experiment( data_hist, categories=self.categories )
    ###
    
    # >>>>>>>>>>>>>>>
    #  CLEAN HIST  #
    # >>>>>>>>>>>>>>>

    def clean_hist( self ) :
        ''' Clean the attribute <data_hist>. ''' 
        self.assign_hist( pd.Series() )
    ###
    
    # >>>>>>>>>>>>>>>>>>>
    #  ASSIGN FEATURES  #
    # >>>>>>>>>>>>>>>>>>>

    def assign_features( self, skip=None, beg=None, end=None ):     
        ''' assign skip, beg, end.'''

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

        self.beg = beg     
        self.end = end
        self.skip = skip

    def _inSequence( self, thisSeq ):
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

        first_idx = self.beg
        last_idx = len( thisSeq ) - self.end - self.num
        
        # mask out of alphabet characters
        thisSeq = ''.join([i if i in self.alphabet else '*' for i in thisSeq ])
        results = [ thisSeq[ i : self.num+i ] for i in range ( first_idx , 1+last_idx , 1+self.skip ) ]
        return results

    def _extract( self, sequences ):
        '''UPDATE DICTIONARY from SEQUENCES'''

        results = list(map(lambda x : self._inSequence( x ), sequences ) )
        return results

    def encode( self, sequences ) :
        ''' Encode all ngrams observed in the respective sequences.'''

        word_length = self.num
        
        # WARNING!: this is a bottleneck
        all_possible_words = list(map( ("").join, product( self.alphabet, repeat=word_length) ) )
        word_dict = dict(zip(all_possible_words, np.arange(len(all_possible_words))))

        extracted_ngrams = self._extract( sequences )
        results = list( map( lambda Words : [word_dict.get(w, -1) for w in Words], extracted_ngrams ) )

        return results
        

    def hist_update( self, sequences, file_output=None ):
        '''
        It updates data hist computing ngrams on each entry of the list "sequences".
        Be careful : there is no control on sequences 

        Parameters
        ----------
        sequences: list
                the list of sequences from which ngrams are extracted.
               
        file_output: path/to/file.csv.gz, optional
                the path/to/file.csv.gz where to save the output (otherways no file is produced). 
        '''

        results = self._extract( sequences )
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


def decode_ngrams( encoded_word_list, code_dict, word_length ) :
    '''Decodes ngrams according to the given code_dict.'''
    # FIXME :

    assert np.max(encoded_word_list) < np.power(len(code_dict),word_length)
    converter = np.power(len(code_dict), np.arange(word_length))[::-1]
    inv_code = {v: k for k, v in code_dict.items()}

    # reconstruct the digitizzed ngram matrix
    tmp = np.array(encoded_word_list)
    ngram_mtx_upside_down = []
    for idx in np.arange(word_length) :
        div = converter[-idx-1]
        ngram_mtx_upside_down.append( np.floor( tmp / div ).astype(int) )
        tmp = np.mod(tmp, div)
    ngram_mtx = np.flipud(ngram_mtx_upside_down)
    # convert back from digits to alphabet
    alph_mtx = np.vectorize(lambda k : inv_code[k])(ngram_mtx)

    return [ ('').join(x) for x in alph_mtx.T ]

#######################
#  COUNTS GENERATORS  #
#######################

def data_generator(
    counts_hist_gen_, size, *chg_args,
    seed=None, thres=1e3, njobs=None,
     ):
    '''Counts generator parallelizer for function counts_hist_gen_( seed, size, *chg_args,).'''

    if njobs is None :
        CPU_count = min( int(np.ceil(size/thres)), multiprocessing.cpu_count() ) 
        # FIXME : does this make sense?

    if CPU_count > 1 :
        size_per_job = np.floor( size / CPU_count ).astype(int)
        number_of_jobs = np.floor( size / size_per_job ).astype(int)
        n_pool = ( size_per_job * np.ones( number_of_jobs ) ).astype(int)
        n_pool[-1] = size - np.sum(n_pool[:-1])

        if seed is None :
            rng = np.random.default_rng(seed)
            seed = rng.integers(1e9)
        child_seeds = np.random.SeedSequence(seed).spawn(len(n_pool))
        args = [ ( s, n, *chg_args ) for s,n in zip(child_seeds, n_pool) ]
        #args = tqdm( args, total=len(args), desc='Generating counts', disable=(not verbose) )
        # FIXME : tqdm update fills the bar instantly 
        POOL = multiprocessing.Pool( CPU_count )
        results = POOL.starmap( counts_hist_gen_, args )
        POOL.close()

        output = results[0]
        for to_add in results[1:] :
            output = output.add(to_add, fill_value=0)
        # NOTE : is it the addition casting to floats?
        output = output.astype(int)
    else :
        output = counts_hist_gen_( seed, size, *chg_args )
    return output


def pmf_data_hist_gen( pmf, size=1, is_counts=True, seed=None ) :
    '''Counts hist generator from the probability mass function'''
    rng = np.random.default_rng( seed )

    sequences = rng.choice( 1+np.arange(len(pmf)), size=size, replace=True, p=pmf  )
    tmp = pd.Series( sequences ).astype(int)
    if is_counts is True :
        output = tmp.groupby( tmp ).size()
    else :
        output = tmp
    return output

######################
#  MACHINE LEARNING  #
######################

def ngram_1hot_encoder(
    sequences, gear=None, 
    neglect_rep=True, drop_non_valid_categ=False, length_normalize=False, include_length=False
    ) :
    ''' 1-hot encoding of the sequences according to their n-gram'''

    if gear is None :
        raise IOError("Specify a ngram model.")
    elif type(gear) != ngram_gear :
        raise IOError("model must be of the class `ngram_gear`.")

    data = gear.encode( sequences )

    # Number of feautures associated to each sequence according to the model
    # plus 1 for the "not-valid-categories"
    input_size = gear.categories + 1
    data_enc = np.zeros( ( len(sequences), input_size ), dtype=np.int8 )
    for i in range( len(sequences) ) :
        if neglect_rep is True :
            data_enc[i][ data[i] ] = 1
        else :
            for c in data[i] : 
                data_enc[i][c] += 1

    # drop non valid categoriers
    if drop_non_valid_categ :
        data_enc = data_enc[:, :-1]

    # normalize according to the length 
    if length_normalize is True :
        lengths = np.array(list(map(lambda x : len(x), sequences)))
        data_enc = data_enc / lengths[:,np.newaxis]

    # add the length information
    if include_length is True :
        lengths = np.array(list(map(lambda x : len(x), sequences)))
        data_enc = np.concatenate( (data_enc, lengths[:,np.newaxis]), axis=1 )

    return data_enc

class ClassifyOnNgrams( object ):
    ''' Regression class for ngram features model. '''

    def __init__(
        self,
        Gear=None, num=None, alph=None, skip=None, beg=None, end=None,
        neglect_rep=True, drop_non_valid_categ=False, length_normalize=False, 
        include_length=False,
        ) :
        
        # Load ngram Model #
        if type(Gear) == ngram_gear :
            self.gear = Gear
        else :
            self.gear = ngram_gear( num=num, alph=alph, skip=skip, beg=beg, end=end )
        self.input_size = self.gear.categories
        self.input_size += int(drop_non_valid_categ is False)
        self.input_size += int(include_length is True)

        # Define Encoder #
        self.encoder = FunctionTransformer(
            ngram_1hot_encoder,
            kw_args={
                "gear": self.gear,
                "neglect_rep": neglect_rep,
                "drop_non_valid_categ": drop_non_valid_categ,
                "length_normalize" : length_normalize,
                "include_length" : include_length
                }
            )
###