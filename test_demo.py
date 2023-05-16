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

import sys
sys.path.append("./")

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
embed._load_model()

bert = BERT(intent_train,intent_test)
bert.load_model()

entity = EntityRecognizer(
    model=LSTM(dataset.entity_dict),
    loss=CRFLoss(dataset.entity_dict)
)
entity._load_model()

chatString="1분 30초 타이머"

data = [chatString, '0']
data = [data]
data = pd.DataFrame(data)

intent_result = bert.predict(data)

text = dataset.prep.tokenize(chatString, train=False)
prep = dataset.load_predict(chatString, embed)
entity_result = entity.predict(prep)

#entity = entity.tolist()

print(intent_result)
print(entity_result)
print(text)


