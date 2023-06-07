# -*- coding: utf-8 -*-
"""
preprocessing label_target.py
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from datetime import datetime as dt

### Helper Functions

def find_sim_single_summary(summary_sentence_embed, doc_emedding):
    '''returns array of indices for max cosine sim per summary sentences'''
    cos_sim_mat = cosine_similarity(doc_emedding, summary_sentence_embed)
    idx_arr = np.argmax(cos_sim_mat, axis=0)
    
    return idx_arr

def label_sent_in_summary(s_text, s_summary):
    '''returns index list and binary target labels in an array'''
    doc_num = s_text.shape[0]
    
    #initialize zeros
    labels = [np.zeros(doc.shape[0]) for doc in s_text.tolist()] 
    
    #calc idx for most similar
    
    idx_list = [np.sort(find_sim_single_summary(s_summary[j], s_text[j])) for j 
                                                            in range(doc_num)]
      
    for j in range(doc_num):
        labels[j][idx_list[j]]= 1 
    
    return idx_list, labels


### Script

t1 = dt.now()
print(t1)

output_file = 'train_stats_df_processed_extr_label_5000.pickle'

df = pd.read_pickle('train_stats_df_processed_extr_5000.pickle' )

#get index list and target labels
idx_list, labels = label_sent_in_summary(df.text_embedding, df.summary_embedding)

#wrap in dataframe
df['labels'] = labels
df['labels_idx_list'] = idx_list

#save to pickle
with open(output_file, 'wb') as handle:                                     
    pickle.dump(df, handle)

t2 = dt.now()

print(t2)

print(t2-t1)


