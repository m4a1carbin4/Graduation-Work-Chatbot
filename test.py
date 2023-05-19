import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MatrixFactorization():
    def __init__(self, R, k, learning_rate, reg_param, epochs, verbose=False):
        """
        :param R: 전체 원본 레이팅 매트릭스 (사용자 X 아이템 리뷰점수 테이블)
        :param k: 잠재벡터 파라미터 (사실상 얼마만큼의 데이터 즉 데이터 차원수를 사용하는가에 대한 파라미터)
        :param learning_rate: 학습률
        :param reg_param: 가중치의 정규화 값
        :param epochs: 학습 횟수
        :param verbose: 상태출력
        """
        self._R = R
        self._num_users, self._num_items = R.shape
        self._k = k
        self._learning_rate = learning_rate
        self._reg_param = reg_param
        self._epochs = epochs
        self._verbose = verbose


    def fit(self):
        """
        training Matrix Factorization : Update matrix latent weight and bias

        참고: self._b에 대한 설명
        - global bias: input R에서 평가가 매겨진 rating의 평균값을 global bias로 사용
        - 정규화 기능. 최종 rating에 음수가 들어가는 것 대신 latent feature에 음수가 포함되도록 해줌.

        :return: training_process
        """

        # init latent features
        self._User_Latent = np.random.normal(size=(self._num_users, self._k))
        self._Item_Latent = np.random.normal(size=(self._num_items, self._k))

        # init biases
        self._b_User_Latent = np.zeros(self._num_users)
        self._b_Item_Latent = np.zeros(self._num_items)
        self._b = np.mean(self._R[np.where(self._R != 0)])

        # train while epochs : 설정된 횟수만큼 반복 학습.
        self._training_process = []
        for epoch in range(self._epochs):
            # rating이 존재하는 index를 기준으로 training
            X_index, Y_index = self._R.nonzero()
            for x, y in zip(X_index, Y_index):
                self.gradient_descent(x, y, self._R[x, y])
            cost = self.cost()
            self._training_process.append((epoch, cost))

            # print status
            if self._verbose == True and ((epoch + 1) % 5 == 0):
                print("Iteration: %d ; cost = %.4f" % (epoch + 1, cost))


    def cost(self):
        """
        compute root mean square error
        :return: rmse cost
        """

        # Nx_index, Ny_index: R[Nx_index, Ny_index]는 nonzero인 value를 의미한다.
        # 참고: http://codepractice.tistory.com/90
        Nx_index, Ny_index = self._R.nonzero()
        # predicted = self.get_complete_matrix()
        cost = 0
        for Nx, Ny in zip(Nx_index,Ny_index):
            cost += pow(self._R[Nx, Ny] - self.get_prediction(Nx, Ny), 2)
        return np.sqrt(cost/len(Nx_index))


    def gradient(self, error, x, y):
        """
        gradient of latent feature for GD

        :param error: rating - prediction error
        :param i: user index
        :param j: item index
        :return: gradient of latent feature tuple
        """

        dp = (error * self._Item_Latent[y, :]) - (self._reg_param * self._User_Latent[x, :])
        dq = (error * self._User_Latent[x, :]) - (self._reg_param * self._Item_Latent[y, :])
        return dp, dq


    def gradient_descent(self, x, y, rating):
        """
        graident descent function

        :param x: user index of matrix
        :param y: item index of matrix
        :param rating: rating of (x,y)
        """

        # get error
        prediction = self.get_prediction(x, y)
        error = rating - prediction

        # update biases
        self._b_User_Latent[x] += self._learning_rate * (error - self._reg_param * self._b_User_Latent[x])
        self._b_Item_Latent[y] += self._learning_rate * (error - self._reg_param * self._b_Item_Latent[y])

        # update latent feature
        dp, dq = self.gradient(error, x, y)
        self._User_Latent[x, :] += self._learning_rate * dp
        self._Item_Latent[y, :] += self._learning_rate * dq


    def get_prediction(self, x, y):
        """
        get predicted rating: user_x, item_y
        :return: prediction of R_xy
        """
        # 전체평균 + User 평균 평점 + item 평균 평점 + User latent X item latent 로 계산된 평점 -> 분산 제거 목적.
        return self._b + self._b_User_Latent[x] + self._b_Item_Latent[y] + self._User_Latent[x, :].dot(self._Item_Latent[y, :].T)


    def get_complete_matrix(self):
        """
        computer complete matrix PXQ + P.bias + Q.bias + global bias

        - PXQ 행렬에 b_P[:, np.newaxis]를 더하는 것은 각 열마다 bias를 더해주는 것
        - b_Q[np.newaxis:, ]를 더하는 것은 각 행마다 bias를 더해주는 것
        - b를 더하는 것은 각 element마다 bias를 더해주는 것

        - newaxis: 차원을 추가해줌. 1차원인 Latent들로 2차원의 R에 행/열 단위 연산을 해주기위해 차원을 추가하는 것.

        :return: complete matrix R^
        """
        return self._b + self._b_User_Latent[:, np.newaxis] + self._b_Item_Latent[np.newaxis:, ] + self._User_Latent.dot(self._Item_Latent.T)

def print_predict(dataset):
    # 해당 column의 이름을 정해서 삭제 하고 싶은 부분을 조절 가능
    dataset = dataset.drop(['Unnamed: 6','Predictions'], axis=1)
    return dataset

def print_predict_list(dataset):
    # 위의 출력부분과 마찬가지의 파트
    dataset = dataset.drop(['재료','조리법'], axis=1)
    return dataset

def create_soup(x):
    return  x['레시피id'] + ' ' + x['카테고리']

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

# 추천시스템 함수 (default : explanation)
def get_recommendations(title,request_num,cosine_sim,indices):
    
    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:request_num+1]
    food_indices = [i[0] for i in sim_scores]
    return df_recipe['레시피id'].iloc[food_indices]

def recommend_recipe(df_svd_preds, user_id, recipe_df, rating_df, num_recommendations=3):
    
    user_idx = df_svd_preds['사용자uid'] == user_id

    sorted_user_predictions = df_svd_preds.loc[user_idx].sort_values(ascending=False)
    user_data = rating_df[rating_df.사용자id == user_id]
    
    user_history = user_data.merge(recipe_df, on = '레시피id').sort_values(['평점'], ascending=False)
    recommendations = recipe_df[~recipe_df['레시피id'].isin(user_history['레시피id'])]
    recommendations = recommendations.merge( pd.DataFrame(sorted_user_predictions).reset_index(), on = '레시피id')
    recommendations = recommendations.rename(columns = {user_row_number: 'Predictions'}).sort_values('Predictions', ascending = False).iloc[:num_recommendations, :]
    
    recommendations = print_predict(recommendations)                  
    recommendations_list = print_predict_list(recommendations) 

    return recommendations_list, recommendations

df_recipe = pd.read_csv('recipe_list.csv')
df_rating = pd.read_csv('user_rating.csv', encoding='cp949')

recipe_rating_matrix = df_rating.pivot(
    index='사용자id',
    columns='레시피id',
    values='평점'
).fillna(0)

rating_indexs = recipe_rating_matrix.index

rating_columns = recipe_rating_matrix.columns

rating_matrix = recipe_rating_matrix.to_numpy()

factorizer = MatrixFactorization(rating_matrix, k=100, learning_rate=0.01, reg_param=0.01, epochs=100, verbose=True)
factorizer.fit()

result = factorizer.get_complete_matrix()

df = pd.DataFrame(result,columns= rating_columns)

df.columns.name="레시피id"

df.insert(0,'사용자uid',rating_indexs)

print(df.head())

user_idx = df['사용자uid'] == 14

print(user_idx)

sorted_user_predictions = df.loc[user_idx].drop(columns='사용자uid')

print(sorted_user_predictions)