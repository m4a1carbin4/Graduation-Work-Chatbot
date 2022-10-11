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

from NER_Classifier import EntityRecognizer
from crfloss import CRFLoss
from NER_LSTM import LSTM


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

data_emb = dataset.load_embed()

intent_train, intent_test = dataset.load_intent()

embed = Embedder(model = Word2Vec(data_emb))

bert = BERT(intent_train,intent_test)

bert.fit()

bert.train_model()

bert.save_model()

data = ['오늘의 서울 날씨 알려줘', '0']
data = [data]
data = pd.DataFrame(data)

result = bert.predict(data)

print(result)

entity = EntityRecognizer(
    model=LSTM(dataset.entity_dict),
    loss=CRFLoss(dataset.entity_dict)
)

entity.fit(dataset.load_entity(embed))
entity._save_model()

prep = dataset.load_predict('오늘의 서울 날씨 알려줘', embed)
entity = entity.predict(prep)

print(entity)