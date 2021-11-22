#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Basket Blind Stat (exec)
    Copyright (C) November 2021 Francesco Camaglia, LPENS 
'''

#####################
#  LOADING MODULES  #
#####################

import os, sys
import glob
import pandas as pd
import numpy as np
import optparse
from tqdm import tqdm

from thymmatu.utils import fileScope, tryMakeDir

##########
#  MAIN  #
##########

def main( ) :
    '''
    Averages everything in the `BASKET`.
    '''
        
    parser = optparse.OptionParser( conflict_handler="resolve" )

    # >>>>>>>>>>>
    #  OPTIONS  #
    # >>>>>>>>>>>
    parser.add_option( '-B', '--BASKET', action='store', dest='BASKET', metavar='path/to/dir', type="string" )
    
    parser.add_option( '-c', '--index_col', action="store", dest='index_col', type=int, default=None )
    parser.add_option( '-h', '--header', action='store', dest='header', default=None  )
    parser.add_option( '-t', '--tag', action="store", dest='tag', default=None )
    parser.add_option( '-s', '--skip', action="store", dest='skiprows', type='int', default=0 )

    # >>>>>>>>>>>>>>>>>>>>>>
    #  OPTIONS ASSIGNMENT  #
    # >>>>>>>>>>>>>>>>>>>>>>

    options, args = parser.parse_args()
    BASKET = options.BASKET
    tag = options.tag
    index_col = options.index_col
    header = options.header
    skiprows = options.skiprows
    
    # >>>>>>>>>>>>>>>
    #  INPUT CHECK  #
    # >>>>>>>>>>>>>>>
    
    #  OUTPUTFILE  
    PathToOut = f"{BASKET}/STAT"
    
    if tag is None : 
        inputfileList = glob.glob( f"{BASKET}/*.*" )
        outputfile = f"{PathToOut}/mean." + ('.').join( inputfileList[0].split('.')[1:] )
        devstdfile = f"{PathToOut}/devStd." + ('.').join( inputfileList[0].split('.')[1:] )
    else : 
        inputfileList = glob.glob( f"{BASKET}/*{tag}*.*" )
        outputfile = f"{PathToOut}/{tag}." + ('.').join( inputfileList[0].split('.')[1:] )
        devstdfile = f"{PathToOut}/{tag}-devStd." + ('.').join( inputfileList[0].split('.')[1:] )
        
    inputfileList = [ f for f in inputfileList if "STAT" not in f ]
    N = len( inputfileList )
    
    if N == 0:
        raise ValueError("Nothing to average on.")
        
    else :
        tryMakeDir( PathToOut )

        # >>>>>>>>>>>>>
        #  EXECUTION  #
        # >>>>>>>>>>>>>       

        #  COMPRESSION & DELIMITER 
        #  infer from extension of first file in list
        in_scope = fileScope( inputfileList[0] )
        delimiter = in_scope.delimiter
        compression = in_scope.compression

        mean_df = pd.read_csv(inputfileList[ 0 ], header=header, sep=delimiter, keep_default_na=False,
                              compression=compression, index_col=index_col, skiprows=skiprows)
        meanPWR2_df = np.power( mean_df, 2 )

        for i in tqdm(range(1, N)) :

            df = pd.read_csv( inputfileList[ i ], header=header, sep=delimiter, keep_default_na=False,
                             compression=compression, index_col=index_col, skiprows=skiprows ) 
            mean_df = mean_df.add(df, fill_value=0)
            meanPWR2_df = meanPWR2_df.add(np.power( df, 2 ),fill_value=0)

        mean_df /= N # by step normalization
        meanPWR2_df /= N # by step normalization

        # sample standard deviation
        devStd_df = np.sqrt( N * (meanPWR2_df - np.power(mean_df, 2)) / (N-1) )

        #  output index  #
        if index_col is None : 
            index = False
        else : 
            index = True   

        #  output header  #
        if header is None : 
            save_header = False
        else : 
            save_header = True

        mean_df.to_csv( outputfile, header = save_header,
                       sep=delimiter, compression=compression, index=index )
        devStd_df.to_csv( devstdfile, header = save_header,
                         sep=delimiter, compression=compression, index=index )

###

################
#  EXECUTABLE  #
################

if __name__ == "__main__" : main( )

