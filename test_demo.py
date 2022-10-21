"""
@auther Hyunwoong
@since 7/1/2020
@see https://github.com/gusdnd852
"""
from data.dataset import Dataset
from gensim.models import Word2Vec
from Settings.decorators import gensim
from Embedder.Word_embedder import Embedder
import pandas as pd
from intent.intent_classification import BERT, BERTClassifier
from NER.NER_Classifier import EntityRecognizer
from base.crfloss import CRFLoss
from NER.NER_LSTM import LSTM

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

bert.data_set()
bert.train_model()

data = ['오늘의 서울 날씨 알려줘', '0']
data = [data]
data = pd.DataFrame(data)

result = bert.predict(data)

print(result)

"""
entity = EntityRecognizer(
    model=LSTM(dataset.entity_dict),
    loss=CRFLoss(dataset.entity_dict)
)

entity.fit(dataset.load_entity(embed))
#entity._load_model()

prep = dataset.load_predict("타이머 3분 맞춰줘", embed)
entity = entity.predict(prep)

#entity = entity.tolist()

print(entity)
"""

