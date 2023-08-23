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

print(movies_df["genres"])
print(movies_df["keywords"])
