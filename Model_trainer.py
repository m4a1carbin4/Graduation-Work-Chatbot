"""
@auther Hyunwoong
@since 7/1/2020
@see https://github.com/gusdnd852
"""
from data.dataset import Dataset
from gensim.models import FastText
from Settings.decorators import gensim
from Embedder.Word_embedder import GensimEmbedder
import pandas as pd

from intent.intent_classification import BERT, BERTClassifier

from NER.NER_Classifier import EntityRecognizer
from base.crfloss import CRFLoss
from NER.NER_LSTM import LSTM


# from scenario import dust, weather, travel, restaurant
# 에러 나면 이걸로 실행해보세요!

@gensim
class FastText(FastText):

    def __init__(self):
        """
        Gensim Word2Vec 모델의 Wrapper 클래스입니다.
        """

        super().__init__(size=self.vector_size,
                         window=self.window_size,
                         workers=self.workers,
                         min_count=self.min_count,
                         iter=self.iter)

dataset = Dataset(False)

data_emb = dataset.load_embed()

intent_train, intent_test = dataset.load_intent()

embed = GensimEmbedder(model = FastText())

embed.fit(data_emb)

bert = BERT(intent_train,intent_test)

bert.data_set()

bert.train_model()

bert.save_model()

data = ['다시한번 들려줘', '0']
data = [data]
data = pd.DataFrame(data)

result = bert.predict(data)

print(result)

entity = EntityRecognizer(
    model=LSTM(dataset.entity_dict),
    loss=CRFLoss(dataset.entity_dict)
)

print(dataset.entity_dict)
entity.fit(dataset.load_entity(embed)) # ram error?
entity._save_model()

prep = dataset.load_predict('타이머 3분 설정해줘', embed)
entity_result = entity.predict(prep)

print(entity_result)
