"""
@auther Hyunwoong
@since 7/1/2020
@see https://github.com/gusdnd852
"""
from dataset import Dataset
from gensim.models import Word2Vec
from decorators import gensim
from Word_embedder import Embedder

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

embed = Embedder(model = Word2Vec(data_emb))

#print(data_emb)

#embed.fit(data_emb)

prep = dataset.load_predict("내일 서울 날씨 어떄?",embed)

print(prep)