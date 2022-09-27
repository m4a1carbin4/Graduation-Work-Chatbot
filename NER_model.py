from cgi import test
from gc import callbacks
import os
from time import time

import os
import shutil
import logging
from tqdm import tqdm, trange

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import tensorflow as tf
from transformers import shape_list, BertTokenizer, TFBertModel
callback = tf.keras.callbacks
sequence = tf.keras.preprocessing.sequence 
from seqeval.metrics import f1_score, classification_report
from transformers import TFBertForTokenClassification
from sklearn.model_selection import train_test_split
import torch
torch.cuda.is_available()

class F1score(tf.keras.callbacks.Callback):
    def __init__(self, X_test, y_test,index_to_tag):
        self.X_test = X_test
        self.y_test = y_test
        self.index_to_tag = index_to_tag

    def sequences_to_tags(self, label_ids, pred_ids):
      label_list = []
      pred_list = []

      for i in range(0, len(label_ids)):
        label_tag = []
        pred_tag = []

        for label_index, pred_index in zip(label_ids[i], pred_ids[i]):
          if label_index != -100:
            label_tag.append(self.index_to_tag[label_index])
            pred_tag.append(self.index_to_tag[pred_index])
        
        label_list.append(label_tag)
        pred_list.append(pred_tag)

      return label_list, pred_list

    def on_epoch_end(self, epoch, logs={}):
      y_predicted = self.model.predict(self.X_test)
      y_predicted = np.argmax(y_predicted.logits, axis = 2)

      label_list, pred_list = self.sequences_to_tags(self.y_test, y_predicted)

      score = f1_score(label_list, pred_list, suffix=True)
      print(' - f1: {:04.2f}'.format(score * 100))
      print(classification_report(label_list, pred_list, suffix=True))

class NER_BERT():
    def __init__(self,dataset,label):

        self.train_data_sentence = [sent.split() for sent in dataset['str'].values]
        self.train_data_label = [tag.split() for tag in dataset['label'].values]

        #self.train_data_sentence,self.train_data_label,self.test_data_sentence,self.test_data_label = train_test_split(self.train_data_sentence,self.train_data_label,test_size=0.2)

        self.labels = [k for k in label.keys()]

        self.tag_to_index = {tag: index for index, tag in enumerate(self.labels)}
        self.index_to_tag = {index: tag for index, tag in enumerate(self.labels)}

        self.tokenizer = BertTokenizer.from_pretrained("klue/bert-base")
        self.tag_size = len(self.tag_to_index)
        self.model = TFBertForTokenClassification.from_pretrained("klue/bert-base", num_labels=self.tag_size, from_pt=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
        self.model.compile(optimizer=self.optimizer)

        self.X_train, self.y_train = self.convert_examples_to_features(self.train_data_sentence, self.train_data_label, max_seq_len=128, tokenizer=self.tokenizer)
        #self.X_test, self.y_test = self.convert_examples_to_features(self.test_data_sentence, self.test_data_label, max_seq_len=128, tokenizer=self.tokenizer)
    
    def train(self):
        self.model.fit(self.X_train, self.y_train, epochs=3, batch_size=32)

    def convert_examples_to_features(self,examples, labels, max_seq_len, tokenizer,
                                 pad_token_id_for_segment=0, pad_token_id_for_label=-100):
        cls_token = tokenizer.cls_token
        sep_token = tokenizer.sep_token
        pad_token_id = tokenizer.pad_token_id

        input_ids, attention_masks, token_type_ids, data_labels = [], [], [], []

        for example, label in tqdm(zip(examples, labels), total=len(examples)):
            tokens = []
            labels_ids = []
            for one_word, label_token in zip(example, label):
                subword_tokens = tokenizer.tokenize(one_word)
                tokens.extend(subword_tokens)
                labels_ids.extend([self.tag_to_index[label_token]]+ [pad_token_id_for_label] * (len(subword_tokens) - 1))

            special_tokens_count = 2
            if len(tokens) > max_seq_len - special_tokens_count:
                tokens = tokens[:(max_seq_len - special_tokens_count)]
                labels_ids = labels_ids[:(max_seq_len - special_tokens_count)]

            tokens += [sep_token]
            labels_ids += [pad_token_id_for_label]
            tokens = [cls_token] + tokens
            labels_ids = [pad_token_id_for_label] + labels_ids

            input_id = tokenizer.convert_tokens_to_ids(tokens)
            attention_mask = [1] * len(input_id)
            padding_count = max_seq_len - len(input_id)

            input_id = input_id + ([pad_token_id] * padding_count)
            attention_mask = attention_mask + ([0] * padding_count)
            token_type_id = [pad_token_id_for_segment] * max_seq_len
            label = labels_ids + ([pad_token_id_for_label] * padding_count)

            assert len(input_id) == max_seq_len, "Error with input length {} vs {}".format(len(input_id), max_seq_len)
            assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
            assert len(token_type_id) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_id), max_seq_len)
            assert len(label) == max_seq_len, "Error with labels length {} vs {}".format(len(label), max_seq_len)

            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            token_type_ids.append(token_type_id)
            data_labels.append(label)

        input_ids = np.array(input_ids, dtype=int)
        attention_masks = np.array(attention_masks, dtype=int)
        token_type_ids = np.array(token_type_ids, dtype=int)
        data_labels = np.asarray(data_labels, dtype=np.int32)

        return (input_ids, attention_masks, token_type_ids), data_labels
    
    