"""
@author : Hyunwoong
@when : 5/9/2020
@homepage : https://github.com/gusdnd852
"""
from gc import callbacks
import os
from time import time

import torch
import numpy as np
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from torch import Tensor

from decorators import gensim
from baseprocessor import BaseProcessor

@gensim
class Embedder(BaseProcessor):
    def __init__(self, model: Word2Vec):
        super().__init__(model)
        self._save_model()

        self.callback = self.GensimLogger(
            name=self.__class__.__name__,
            logging=self._print
        )  # 학습 진행사항 출력 콜백

    def fit(self, dataset : list):
        raise Exception("Word2Vec fit isn't support")

    def predict(self, sequence : str) -> Tensor:
        self._load_model()
        return self._forward(self.model, sequence)
    
    def _load_model(self):
        if not os.path.exists(self.model_dir):
            raise Exception("모델을 불러올 수 없습니다.")
        if not self.model_loaded:
            self.model_loaded = True
            self.model = self.model.__class__.load(self.model_file + '.gensim')
    
    def _save_model(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.model.save(self.model_file +'.gensim')
    
    def _forward(self, model, sequence: str) ->Tensor:
        sentence_vector = []

        for word in sequence:
            try:
                word_vector = torch.tensor(model.wv[word])
            except KeyError as _:
                word_vector = torch.ones(self.vector_size) * self.OOV
            
            sentence_vector.append(word_vector.unsqueeze(dim=0))
        
        return torch.cat(sentence_vector, dim=0)

    #kochat Logger
    class GensimLogger(CallbackAny2Vec):

        def __init__(self, name: str, logging):
            """
            Gensim 모델의 학습 과정을 디버깅하기 위한 callback
            :param name: 모델 이름
            :param print: base processor의 print 함수를 전달받습니다.
            """

            self.epoch, self.eta = 0, 0
            self.name = name
            self.logging = logging

        def on_epoch_begin(self, model: Word2Vec):
            """
            epoch 시작시에 시간 측정을 시작합니다.
            :param model: 학습할 모델
            """
            self.eta = time()

        def on_epoch_end(self, model: Word2Vec):
            """
            epoch 종료시에 걸린 시간을 체크하여 출력합니다.
            :param model: 학습할 모델
            """

            self.logging(
                name=self.name,
                msg='Epoch : {epoch}, ETA : {sec} sec'
                    .format(epoch=self.epoch, sec=round(time() - self.eta, 4))
            )

            self.epoch += 1

    