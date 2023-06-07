# -*- coding: utf-8 -*-
"""
filter_extractive_data_only.py
""" 
import pickle
import pandas as pd
 
output_file = 'train_stats_df_extractive_no_spacy.pickle'

#load all data
df = pd.read_pickle('train_stats_df_no_spacy.pickle')

#filter for extractive summaries only
df = df[df.density_bin == 'extractive']

#save to pickle file
with open(output_file, 'wb') as handle:                                     
    pickle.dump(df, handle)
