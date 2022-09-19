from gc import callbacks
import os
from time import time

import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
from pytorch_transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from torch.optim import Adam
import torch.nn.functional as F

from decorators import gensim, intent
from baseprocessor import BaseProcessor

class BERTDataset(Dataset):
    ''' Naver Sentiment Movie Corpus Dataset '''
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx, 1]
        label = self.df.iloc[idx, 2]
        return text, label

@intent
class BERT():
    def __init__(self) -> None:
        pass

    def fit(self,dataset : list):
        intent_dataset = BERTDataset(dataset)
        train_loader = DataLoader(intent_dataset, batch_size=2, shuffle=True, num_workers=2)
        self.__train(train_loader)
    
    def __train(train_loader):
        device = torch.device("cuda")
        tokenizer = 
