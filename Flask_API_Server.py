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

from flask import Flask, request
from flask_restx import Api
from flask_restx import Resource
import re

def getChatTime(ner):
    if '분' in ner:
        result = int(re.sub(r'[^0-9]','',ner))*60
        
        return result
    elif '초' in ner:
        result = int(re.sub(r'[^0-9]','',ner))

        return result
    
def getPageNum(ner):

    if ner.isalpha():
        if '장' in ner:
            ner = ner[:-1]
        elif '페이지' in ner:
            ner = ner[:-3]

        if ner == "한":
            print("1")
            return 1
        elif ner == "두":
            print("2")
            return 2
        elif ner == "세":
            print("3")
            return 3
        elif ner == "네":
            print("4")
            return 4
        elif ner == "다섯":
            print("5")
            return 5
    elif ner.isdigit():
        return int(ner)
    else :
        if '장' in ner:
            return int(re.sub(r'[^0-9]','',ner))
        elif '페이지' in ner:
            return int(re.sub(r'[^0-9]','',ner))

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

app = Flask(__name__)
api = Api(app)

dataset = Dataset(False)

data_emb = dataset.load_embed()

intent_train, intent_test = dataset.load_intent()

embed = GensimEmbedder(model = FastText())
embed._load_model()

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

        text = dataset.prep.tokenize(chatString, train=False)
        intent_result = bert.predict(data)
        prep = dataset.load_predict(chatString, embed)
        entity_result = entity.predict(prep)

        append = 0

        if intent_result == "0":
            for ner ,txt in zip(entity_result,text):
                if ner != '0':
                    append = append + getChatTime(txt)
            
            print(append)
        elif intent_result == "1":
            for ner ,txt in zip(entity_result,text):
                if ner != '0':
                    append = append + getPageNum(txt)
            if append == 0:
                append = 1
            print(append)
        elif intent_result == "2":
            for ner ,txt in zip(entity_result,text):
                if ner != '0':
                    append = append + getPageNum(txt)
            if append == 0:
                append = 1
            append = append * -1
            print(append)
        elif intent_result == "3":
            append = 0
            for ner in entity_result:
                if ner == 'S-NOW':
                    append = 1
                if ner == 'S-TOTAL':
                    append = 0
                    
            print(append)

        print(entity_result)
        
        return {
            'intent':intent_result,
            'entity':entity_result,
            'append':append,
            'text':text
        }

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8088)