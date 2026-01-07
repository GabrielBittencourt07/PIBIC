# ------------------------------------------------------------------------- #
#                                   Imports                                 #
# ------------------------------------------------------------------------- #

import os
import pandas as pd
import numpy as np
from datetime import timedelta
from scipy import signal
import tsfel
import re


# ------------------------------------------------------------------------- #
#                                  Functions                                #
# ------------------------------------------------------------------------- #

def transitions(df):
    """
    Processes df_raw to create new columns:
        1. 'Trans-Pos' - Transition in Position for consecutive positions performed by same dog
        2. 'Trans-Time' - Transition in Time in case moving in between two positions or different dog
        3. 'Transition' -  Transition column combining both step 1. and 2.
    Parameters
    ----------
    """
    # Finding transitions in posture
    df['Trans-Pos'] = df['Position'].shift()  != df['Position']
    # Finding transitions in time that are bigger than the 100Hz -> 10,000 microseconds
    df['Trans-Time'] = (df["Timestamp"].diff() != timedelta(microseconds = 10000)) # + \

    # Combining the time and position transitions
    df['Transition'] = df['Trans-Pos'] + df['Trans-Time']
    # Changing last row into a transition, Transition column has s_idx and f_idx of the BT
    df.loc[df.index[-1], 'Transition'] = True

    return(df)


def features_tsfel(df_raw, w_size, w_overlap, t_time, n_jobs=4):
    """
    Extrai características usando TSFEL (Time Series Feature Extraction Library)
    com suporte a processamento paralelo.

    Parameters
    ----------
    df_raw (DataFrame): DataFrame com os dados brutos
    w_size (int): Tamanho da janela para extração de características
    w_overlap (float): Sobreposição entre janelas consecutivas
    t_time (timedelta): Passo de tempo entre amostras
    n_jobs (int): Número de jobs para processamento paralelo (default: 4)

    Returns
    ----------
    df_feat (DataFrame): DataFrame com as características extraídas
    """
    df_feat = None

    # Finding transitions in posture, adds 3 more columns to original dataframe
    df_raw = transitions(df_raw)

    # Selecting sensor columns with X., Y., or Z.
    cols = df_raw.columns[df_raw.columns.str.contains(r'\.X|\.Y|\.Z', regex=True)]

    # feature extraction settings
    cfg_full = tsfel.get_features_by_domain()

    cfg = {
        "statistical": {
            "Mean": cfg_full["statistical"]["Mean"],
            "Standard deviation": cfg_full["statistical"]["Standard deviation"],
        }
    }

    for subject in df_raw['Subject'].unique():

        # selecting dog and dc
        df = df_raw[df_raw['Subject'] == subject].reset_index(drop=True)

        # list to add tsfel dataframes
        df_list = []
        # Start and Finish Index for each position
        s_indices = df.index[df['Transition'] == True][:-1]
        f_indices = df.index[df['Transition'].shift(-1) == True]

        # Iterating over the postures, taking the steady periods between transitions
        for (s_idx, f_idx) in zip(s_indices, f_indices):

            if(df.loc[s_idx:f_idx].shape[0] >= w_size):

                df_list.append(tsfel.time_series_features_extractor(
                                # configuration file with features to be extracted
                                config = cfg,
                                # dataframe window to calculate features window on
                                timeseries = df.loc[s_idx:f_idx, cols],
                                # name of header columns
                                header_names = cols,
                                # sampling frequency of original signal
                                fs = 100,
                                # sliding window size
                                window_size = w_size,
                                # overlap between subsequent sliding windows
                                overlap = w_overlap,
                                # do not create a progress bar
                                verbose = 0,
                                # número de jobs para processamento paralelo
                                n_jobs = n_jobs).assign(
                                Timestamp = df.loc[s_idx, 'Timestamp'],
                                Subject = df.loc[s_idx,'Subject'],
                                Position = df.loc[s_idx, 'Position']))


        df_feat = pd.concat(df_list)
        df_feat.set_index('Timestamp', inplace = True)

    return(df_feat)
