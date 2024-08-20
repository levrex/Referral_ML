#import numpy as np
import multiprocessing as mp
import time
import concurrent.futures
import pandas as pd
import re
import os
from flair.data import Sentence
from pathlib import Path
import flair
from flair.models import SequenceTagger
import pandas as pd
from flair.embeddings import TransformerWordEmbeddings
from torch import nn
import torch
import pickle as pl
import logging

print('test')
logging.warning('Watch out!')  # will print a message to the console

flair.cache_root = Path("/exports/reum/tdmaarseveen/cache/")# WORKS

os.environ["TRANSFORMERS_CACHE"] = "/exports/reum/tdmaarseveen/cache/"
os.environ["DATASETS_CACHE"] = "/exports/reum/tdmaarseveen/cache/"
os.environ["HF_TRANSFORMERS_CACHE"] = "/exports/reum/tdmaarseveen/cache/"

tagger = SequenceTagger.load("flair/ner-dutch")
embeddings = TransformerWordEmbeddings('GroNLP/bert-base-dutch-cased')
new_embedding_tensor = torch.cat([tagger.embeddings.model.get_input_embeddings().weight, embeddings.model.get_input_embeddings().weight[tagger.embeddings.model.get_input_embeddings().num_embeddings:-1]])
new_input_embeddings = nn.Embedding.from_pretrained(new_embedding_tensor) 
tagger.embeddings.model.set_input_embeddings(new_input_embeddings)
tagger.embeddings.base_model_name="GroNLP/bert-base-dutch-cased"

logging.warning('Watch out2!')  # will print a message to the console
df = pd.read_csv('/exports/reum/tdmaarseveen/gitlab/referral_ml/proc/ZWN_referral_proc.csv', sep='|')

# Drop letters that were after first visit
df = df[((df['before_firstVisit']))].reset_index(drop=True)

def parallelize_dataframe(df, func):
    set_entities = []
    num_processes = mp.cpu_count()
    print('Num processes:' + str(num_processes))
    df_split = np.array_split(df, num_processes)
    with mp.Pool(num_processes) as p:
        print('DF length:', len(df_split))
        
        df = pd.concat(p.map(parallelize_function, df_split))
    return df

def parallelize_function(df):
    return df['proc_RTFscripted'].apply(screen_entities)

def screen_entities(x):
    sent = x.replace('.', '').replace(')', '').replace('(', '').replace(',', '')
    # .replace('(', ' ( ').replace(')', ' ) ').replace('\\', ' \\ ').replace('[', ' [ ').replace(']', ' ] ')
    sent = sent.replace('  ', ' ')
    thresh = .8
    # make example sentence
    sentence = Sentence(sent)
    # predict NER tags
    tagger.predict(sentence)
    d_tag = {'PER' :  'PERSON', 'LOC' :  'LOCATION'}
    l_ent = [[entity.text, entity.score, entity.tag] for entity in sentence.get_spans('ner') if (entity.tag in ['PER', 'LOC'] and len(entity.text)> 1)]
    transformer_o = [(entity.text) for entity in sentence.get_spans('ner') if (entity.tag in ['PER', 'LOC'] and entity.score > thresh and len(entity.text)> 1)]
    
    # Assume that Named entities never contain special characters : such as brackets or 
    # Regular expression pattern to match special characters
    qualified = re.compile(r'[^a-zA-Z0-9\s]')
    # Remove special characters from each string in the list
    cleaned_list = [qualified.sub('', s) for s in list(set(transformer_o))]
    
    # Regex list
    if len(cleaned_list) > 2:
        re_pat = '(' + ")|(".join(cleaned_list) + ')'
    elif len(cleaned_list) > 1:
        re_pat = '(' + cleaned_list[0] + ')'
    else:
        re_pat = '()'
    print(len(re_pat), flush=True)
    # Redact information
    if re_pat != '()': 
        try:
            fix_line = re.sub(re_pat, ' [REDACTED] ', sent)
        except: 
            print('Error', re_pat)
            print(sent)
            fix_line = sent
            print(eqokte)
    else :
        fix_line = sent
    return str(transformer_o) + '  ' + str(l_ent) + '  ' + str(fix_line) 

#df = df_referral.copy()#.head(10) #df_unfold#.head(5)

start_time = time.time()

df['NER'] = df['proc_RTFscripted'].apply(lambda x : screen_entities(x)) #parallelize_dataframe(df, screen_entities)

# Monkey fix
df[['Entities', 'Annotation', 'FixedLine']] = df['NER'].str.split('  ', n=2, expand=True)

# Record the end time
end_time = time.time()

# Export DF
df[['Identifier', 'RA', 'FirstDiagnosis', 'FirstVisit',
       'referral_date', 'delta_referral_diagnosis',
       'delta_referral_visit', 'before_diagnosis', 'before_firstVisit',
       'Entities', 'Annotation', 'FixedLine']].to_csv('/exports/reum/tdmaarseveen/gitlab/referral_ml/proc/ZWN_referral_proc_blinded.csv', sep='|')

print(end_time - start_time)
