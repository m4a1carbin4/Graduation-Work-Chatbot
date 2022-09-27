"""
@auther Hyunwoong
@since 7/1/2020
@see https://github.com/gusdnd852
"""
from dataset import Dataset
from gensim.models import Word2Vec
from decorators import gensim
from Word_embedder import Embedder
import pandas as pd

from intent_classification import BERT, BERTClassifier
from NER_train import Trainer
from NER_model import NER_BERT

# from scenario import dust, weather, travel, restaurant
# 에러 나면 이걸로 실행해보세요!

@gensim
class Word2Vec(Word2Vec):

    def __init__(self,data:list):
        """
        Gensim Word2Vec 모델의 Wrapper 클래스입니다.
        """

        super().__init__(sentences = data, vector_size=self.vector_size,
                         window=self.window_size,
                         workers=self.workers,
                         min_count=self.min_count)

dataset = Dataset()

print(dataset.intent_dict)

print(dataset.entity_dict)

data_emb = dataset.load_embed()

intent_train, intent_test = dataset.load_intent()

embed = Embedder(model = Word2Vec(data_emb))

#bert = BERT(intent_train,intent_test)

#bert.fit()

#bert.train_model()

#bert.load_model()
"""
data = ['오늘의 서울 날씨 알려줘', '0']
data = [data]
data = pd.DataFrame(data)
"""

#result = bert.predict(data)

#print(result)

NER_data = dataset.load_entity_bert()
"""
NER = NER_BERT(NER_data,dataset.entity_dict)

NER.train()
ent_train, ent_test ,train_label, test_label = dataset.load_entity(embed)
"""
ner = Trainer(NER_data,dataset.entity_dict)

ner.train()

#print(data_emb)

#embed.fit(data_emb)