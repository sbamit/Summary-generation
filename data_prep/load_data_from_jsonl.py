# -*- coding: utf-8 -*-
"""
load_data_from_jsonl.py
"""
import json
import pandas as pd
import pickle

input_file = 'train-stats.jsonl'
output_file = 'train_stats_df_no_spacy.pickle'

#read jsonl file into list of sample rows
counter=0
data=[]
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line.rstrip('\n|\r')))
        counter +=1
        if counter > 500000 :
          break
print('number of lines:', counter)
#wrap in dataframe        
df = pd.DataFrame(data)

#save to pickle
with open(output_file, 'wb') as handle:                                     
    pickle.dump(df, handle)

