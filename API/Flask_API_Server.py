"""
@auther Hyunwoong
@since 7/1/2020
@see https://github.com/gusdnd852
"""
from data.dataset import Dataset
from gensim.models import Word2Vec
from Settings.decorators import gensim, intent
from Embedder.Word_embedder import Embedder
import pandas as pd

from intent.intent_classification import BERT, BERTClassifier

from NER.NER_Classifier import EntityRecognizer
from base.crfloss import CRFLoss
from NER.NER_LSTM import LSTM

from flask import Flask, request
from flask_restx import Api
from flask_restx import Resource
import json

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

app = Flask(__name__)
api = Api(app)

dataset = Dataset()

data_emb = dataset.load_embed()

intent_train, intent_test = dataset.load_intent()

embed = Embedder(model = Word2Vec(data_emb))

bert = BERT(intent_train,intent_test)
bert.load_model()

entity = EntityRecognizer(
    model=LSTM(dataset.entity_dict),
    loss=CRFLoss(dataset.entity_dict))
entity._load_model()

@api.route('/chat_request')
class request_chat(Resource):

    def post(self):
        content = request.get_json()

        print(content)

        chatString = content['chatString']

        data = [chatString, '0']
        data = [data]
        data = pd.DataFrame(data)

        intent_result = bert.predict(data)

        print(intent_result)

        prep = dataset.load_predict(chatString, embed)
        entity_result = entity.predict(prep)

        print(entity_result)

        return {
            'chatString':chatString,
            'intent':intent_result,
            'entity':entity_result
        }

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)