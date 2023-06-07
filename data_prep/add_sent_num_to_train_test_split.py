# -*- coding: utf-8 -*-
"""
add_sent_num_to_train_test_split.py
"""
import pickle
import pandas as pd
from functions import gen_train_test_split_doc_level


input_filename = 'extractive_all_domain_labels.pickle' 
folds = 1

output_file = 'train_test_set20_embeddings_sent_num.pickle'

data_dict = pd.read_pickle(input_filename)

#Specify model inputs: df, X, y, doc_labels
df = data_dict['df_original']
Xy_doc_label = data_dict['Xy_doc_label_array']
X = data_dict['df_X'].drop(['Doc_Length'], axis=1).values
y = data_dict['y_array']

        
#train test split at document level

train_test_set = gen_train_test_split_doc_level(Xy_doc_label, X, y, 
                                         test_ratio=0.2, folds=folds, rand_seed=42)

data_dict.update({'train_test_sets': train_test_set })

with open(output_file, 'wb') as handle:                                     
    pickle.dump(data_dict, handle)

