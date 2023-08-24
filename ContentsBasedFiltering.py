# 추천 기능
# 컨텐츠 기반 필터링 - Contents Based Filtering
# 방법
#   1. 영화 구성 컨텐츠 텍스트
#   2. 피쳐 백터화(Count, TF-IDF)
#   3. 코사인 유사도
#   4. 유사도 및 평점에 따른 영화 추천
# 순서
#   1. 컨텐츠에 대한 여러 텍스트 정보들을 피쳐 벡터화하기
#   2. 코사인 유사도로 컨텐츠 별 유사도 계산
#   3. 컨텐츠 별로 가중 평점 계산
#   4. 유사도가 높은 컨텐츠 중에 평점이 좋은 컨텐츠 순으로 추천

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from ast import literal_eval

pd.set_option("display.max_columns", 100)
pd.set_option("display.max_colwidth", 200)

movies = pd.read_csv("data/tmdb_5000_movies.csv")
movies_row_before_processing = movies.shape[0]
movies_col_before_processing = movies.shape[1]
print(movies_row_before_processing, movies_col_before_processing)

movies_df = movies[["id",
                    "title",
                    "genres",
                    "vote_average",
                    "vote_count",
                    "popularity",
                    "keywords",
                    "overview"]]
print(movies_df.info())


def returnNames(x):
    names = []
    for y in x:
        names.append(y["name"])

    return names


movies_df["genres"] = movies_df["genres"].apply(literal_eval)
movies_df["keywords"] = movies_df["keywords"].apply(literal_eval)
movies_df["genres"] = movies_df["genres"].apply(lambda x: returnNames(x))
movies_df["keywords"] = movies_df["keywords"].apply(lambda x: returnNames(x))

# print(movies_df["genres"])
# print(movies_df["keywords"])
# print(movies_df[["genres", "keywords"]][:1])

# 장르 컨텐츠 필터링을 이용한 영화 추천. 장르 문자열을 Count 벡터화 후에 코사인 유사도로 각 영화를 비교
# 장르 문자열의 Count기반 픽쳐 벡터화
from sklearn.feature_extraction.text import CountVectorizer

# CountVectorizer를 적용하기 위해 공백 문자로 word 단위가 구분되는 문자열로 변환
movies_df["genres_literal"] = movies_df["genres"].apply(lambda x : (" ").join(x))
count_vect = CountVectorizer(min_df=0, ngram_range=(1,2))
genre_mat = count_vect.fit_transform(movies_df["genres_literal"])
print(genre_mat.shape)

# 장르에 따른 영화별 코사인 유사도 추출
from sklearn.metrics.pairwise import cosine_similarity

genre_sim = cosine_similarity(genre_mat, genre_mat)
# print(genre_sim.shape)
# print(genre_sim[:2])

genre_sim_sorted_ind = genre_sim.argsort()[:, ::-1]
# print(genre_sim_sorted_ind[:1])

# 특정 영화와 장르별 유사도가 높은 영화를 반환하는 함수
def find_sim_movie(df, sorted_ind, title_name, top_n=10):
    # 인자로 입력된 movies_df 에서 "title" 칼럼이 입력된 title_name 값인 DataFrame 추출
    title_movie = df[df["title"] == title_name]

    # title_name을 가진 DataFrame의 index 객체를 ndarray로 반환하고
    # sorted_ind 인자로 입력된 genre_sim_sorted_ind 객체에서 유사도 순으로 top_n 개의 index 추출
    title_index = title_movie.index.values
    similar_indexes = sorted_ind[title_index, :(top_n)]

    # 추출된 top_n index들 출력 top_n index는 2차원 데이터.
    # DataFrame에서 index로 사용하기 위해 1차원 array로 변경
    print(similar_indexes)
    similar_indexes = similar_indexes.reshape(-1)

    return df.iloc[similar_indexes]


similar_movies = find_sim_movie(movies_df, genre_sim_sorted_ind, "The Godfather", 10)
print(similar_movies[["title", "vote_average"]])

# **문제 : 장르만 같은 영화만 추천함
# **솔루션: 장르가 비슷하고, 평점이 좋은 순으로 영화를 추천

print(movies_df[["title", "vote_average", "vote_count"]].sort_values("vote_average", ascending=False)[:10])

# 평가 횟수에 대한 가중치가 부여된 평점(weighted Rating) 계산
# 가중 평점(Weighted Rating = (v / (v + m))R + (m / (v + m))C
# v : 개별 영화에 평점을 투표한 횟수
# m : 평점을 부여하기 위한 최소 투표 횟수
# R : 개별 영화에 대한 평균 평점
# C : 전체 영화에 대한 평균 평점
C = movies_df["vote_average"].mean()
m = movies_df["vote_count"].quantile(0.6)
print("C:", round(C, 3), "\nm:", round(m, 3))

percentile = .6
m = movies_df["vote_count"].quantile(percentile)
C = movies_df["vote_average"].mean()

def weighted_vote_average(record):
    v = record["vote_count"]
    R = record["vote_average"]

    return (v / (v+m)) * R + (m / (v+m)) * C


movies_df["weighted_vote"] = movies_df.apply(weighted_vote_average, axis=1)
print(movies_df[['title', 'vote_average', 'weighted_vote', 'vote_count']].sort_values("weighted_vote", ascending=False)[:10])

def find_sim_movie(df, sorted_ind, title_name, top_n=10):
    title_movie = df[df["title"] == title_name]
    title_index = title_movie.index.values

    # top_n의 2배에 해당하는 장르 유사성이 높은 index 추출
    similar_indexes = sorted_ind[title_index, :(top_n*2)]
    similar_indexes = similar_indexes.reshape(-1)

    # 기준 영화 index 제외
    similar_indexes = similar_indexes[similar_indexes != title_index]

    # top_n의 2배에 해당하는 후보군에서 weighted_vote 높은 순으로 top_n만큼 추출
    return df.iloc[similar_indexes].sort_values("weighted_vote", ascending=False)[:top_n]

similar_movies = find_sim_movie(movies_df, genre_sim_sorted_ind, "The Godfather", 10)
print(similar_movies[["title", "vote_average", "weighted_vote"]])