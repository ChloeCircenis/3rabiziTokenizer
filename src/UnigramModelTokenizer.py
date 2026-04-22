import numpy as np
import os 
import pandas as pd
import kagglehub
import sentencepiece as spm

class UnigramModelTokenizer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
    def tokenize(self, text):
        return self.sp.encode(text, out_type=str)
    def detokenize(self, tokens):
        return self.sp.decode(tokens)

def train_unigram_model(data_path, model_prefix, vocab_size=30000):
    df = pd.read_csv(data_path)
    texts = df['text'].tolist()
    
    temp_file = 'temp_texts.txt'
    with open(temp_file, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + '\n')
    
    spm.SentencePieceTrainer.Train(
        input=temp_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type='unigram'
    )
    
    os.remove(temp_file)
    model_path = f'{model_prefix}.model'
    return model_path
