from cgi import test
from gc import callbacks
import os
from time import time

import torch
import pandas as pd
import numpy as np

from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder

from Settings.decorators import gensim, intent
from base.baseprocessor import BaseProcessor

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)
        
        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes = 4, # softmax 사용 <- binary일 경우는 2
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

@intent
class BERT():
    def __init__(self,train:list,test:list):

        self.device = torch.device("cuda:0")
        self.bertmodel, self.vocab = get_pytorch_kobert_model()
        self.train = train
        self.test = test
        self.model_loaded = False
        
        self.model = BERTClassifier(self.bertmodel, dr_rate=0.5).to(self.device)
        self.tokenizer = get_tokenizer()
        self.tok = nlp.data.BERTSPTokenizer(self.tokenizer, self.vocab, lower=False)

    def data_set(self):

        data_train = BERTDataset(self.train, 0, 1, self.tok, self.max_len, True, False)
        data_test = BERTDataset(self.test, 0, 1, self.tok, self.max_len, True, False)

        self.train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=self.batch_size, num_workers=5)
        self.test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=self.batch_size, num_workers=5)
    
    def train_model(self):

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        # 옵티마이저 선언
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        loss_fn = nn.CrossEntropyLoss() # softmax용 Loss Function 정하기 <- binary classification도 해당 loss function 사용 가능

        t_total = len(self.train_dataloader) * self.num_epochs
        warmup_step = int(t_total * self.warmup_ratio)

        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

        # 학습 평가 지표인 accuracy 계산 -> 얼마나 타겟값을 많이 맞추었는가
        def calc_accuracy(X,Y):
            max_vals, max_indices = torch.max(X, 1)
            train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
            return train_acc
        
        # 모델 학습 시작
        for e in range(self.num_epochs):
            train_acc = 0.0
            test_acc = 0.0
            
            self.model.train()
            for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(self.train_dataloader)):
                optimizer.zero_grad()
                token_ids = token_ids.long().to(self.device)
                segment_ids = segment_ids.long().to(self.device)
                valid_length= valid_length
                label = label.long().to(self.device)
                out = self.model(token_ids, valid_length, segment_ids)
                loss = loss_fn(out, label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm) # gradient clipping
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                train_acc += calc_accuracy(out, label)
                if batch_id % self.log_interval == 0:
                    print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
            print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
            
            self.model.eval() # 평가 모드로 변경
            
            for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(self.test_dataloader)):
                token_ids = token_ids.long().to(self.device)
                segment_ids = segment_ids.long().to(self.device)
                valid_length= valid_length
                label = label.long().to(self.device)
                out = self.model(token_ids, valid_length, segment_ids)
                test_acc += calc_accuracy(out, label)
            print("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))

    def predict(self,input):

        pred = BERTDataset(input.values, 0, 1, self.tok, self.max_len, True, False)
        pred_dataloader = torch.utils.data.DataLoader(pred, batch_size=self.batch_size, num_workers=5)
        
        self.model.eval()

        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(pred_dataloader):
            token_ids = token_ids.long().to(self.device)
            segment_ids = segment_ids.long().to(self.device)

            valid_length= valid_length
            label = label.long().to(self.device)

            out = self.model(token_ids, valid_length, segment_ids)

            test_eval=[]
            for i in out:
                logits=i
                logits = logits.detach().cpu().numpy()

                if np.argmax(logits) == 0:
                    test_eval.append("0")
                elif np.argmax(logits) == 1:
                    test_eval.append("1")
                elif np.argmax(logits) == 2:
                    test_eval.append("2")
                elif np.argmax(logits) == 3:
                    test_eval.append("3")

            print(">> result of input sentence :  " + test_eval[0])

        return test_eval[0]

    def save_model(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.model.state_dict(), self.model_file + '.pt')
    
    def load_model(self):
        if not os.path.exists(self.model_dir):
            raise Exception("모델을 불러올 수 없습니다.")
        if not self.model_loaded:
            self.model_loaded = True
            self.model.load_state_dict(torch.load('koBERT_model_state_dict.pt'))
        
