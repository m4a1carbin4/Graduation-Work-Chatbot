import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel

from flask import Flask, request
from flask_restx import Api
from flask_restx import Resource
import json

#Load Data From File (.tsv)
df = pd.read_csv('category_tsv.tsv',delimiter='\t')

#TF-IDF 불용어 제거
tfidf = TfidfVectorizer(stop_words='english')
# Replace NaN with an empty string
df['food_ex'] = df['food_ex'].fillna('')
#  tfidf matrix 만든다
tfidf_matrix = tfidf.fit_transform(df['food_ex'])

#linear_kernel 한 후 cosine similarity 계산
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# index와 food title의 역방향 matrix 만들기
indices = pd.Series(df.index, index=df['food']).drop_duplicates()

# 추천시스템 함수 (default : explanation)
def get_recommendations(title, cosine_sim=cosine_sim,indices=indices):
    
    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:3]
    food_indices = [i[0] for i in sim_scores]
    return df['food'].iloc[food_indices]

# new DataFrame for category base 
df_new1 = df[['food', 'food_category']]

# 데이터 내의 빈칸 처리
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

# Apply clean_data function to your features.
features = ['food', 'food_category']

for feature in features:
    df_new1[feature] = df_new1[feature].apply(clean_data)

def create_soup(x):
    return  x['food'] + ' ' + x['food_category']

df100 = df_new1.copy()        
df_new1['soup'] = df_new1.apply(create_soup, axis=1)

df_new2 = df_new1
# CountVectorizer를 통해서 진행, 불용어 제거
from sklearn.feature_extraction.text import CountVectorizer
stopwords=['으로', '가', '을', '를', '에', '로', '에게', '또한', ',', '.']

count = CountVectorizer(stop_words=stopwords)
count_matrix = count.fit_transform(df_new2['soup'])

# 다른 cosine_similarity를 구해준다, 이것은 category 에 대한 matrix
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

print(cosine_sim2)

df_new1 = df_new1.reset_index()
indices_c = pd.Series(df_new1.index, index=df_new1['food'])

app = Flask(__name__)
api = Api(app)

@api.route('/chat_request')
class request_chat(Resource):

    def post(self):
        content = request.get_json()

        print(content)

        request_ID = content['last_ID']

        request_type = content['type']

        if(request_type == 0):
            result = get_recommendations(request_ID)
        elif (request_type == 1):
            result = get_recommendations(request_ID, cosine_sim2,indices_c)
        result = result.values.tolist()
        print(result)
        
        return {
            'recommendation_result':result
        }

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)