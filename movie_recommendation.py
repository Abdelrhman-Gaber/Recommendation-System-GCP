import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def read_data(*filepath):
  credit = pd.read_csv(filepath[0])
  movies = pd.read_csv(filepath[1])
  return credit, movies


def merge_data(movies , credit):
  return pd.merge(movies,credit , on = 'title')


def select_columns(df):
  return  df[['movie_id','title','overview','genres','keywords','cast','crew']]


def clean_data(df):
  return df.dropna()


def convert(col):
  lst = []
  for val in ast.literal_eval(col):
    lst.append(val['name'])
  return lst  


def convert_top3(col):
  lst = []
  count = 0
  for val in ast.literal_eval(col):
    if count !=3:
      lst.append(val['name'])
      count +=1
    else:
      break
  return lst


def get_director(col):
  lst=[]
  for val in ast.literal_eval(col):
    if val['job'] =='Director':
      lst.append(val['name'])
      break
  return lst


def preprocess(df):
  cv = CountVectorizer(max_features=5000, stop_words='english')
  return cv.fit_transform(df['tags']).toarray()


def model(vectors):
  return cosine_similarity(vectors)


def recommend(simlarity,new_df ,movie):
  movie_index = new_df[new_df['title']== movie].index[0]
  distance = simlarity[movie_index]
  movie_list = sorted(list(enumerate(distance)),reverse= True,key = lambda x:x[1])[1:6]
  for i in movie_list:
    print(new_df.iloc[i[0]].title)

def save_model(simlarity, new_df):
  pickle.dump(simlarity,open('sim.pkl','wb'))
  pickle.dump(new_df.to_dict(),open('df.pkl','wb'))


def main():
  credit_path = 'tmdb_5000_credits.csv'
  movies_path = 'tmdb_5000_movies.csv'
  credit , movies = read_data(credit_path , movies_path)
  df = merge_data(movies , credit)
  df_new = select_columns(df)
  df_new = clean_data(df_new)
  df_new['genres'] = df_new['genres'].apply(convert)
  df_new['keywords'] = df_new['keywords'].apply(convert)
  df_new['cast'] = df_new['cast'].apply(convert_top3)
  df_new['crew'] = df_new['crew'].apply(get_director)
  df_new['overview'] = df_new['overview'].apply(lambda x :x.split())
  df_new['genres'] = df_new['genres'].apply(lambda x : [i.replace(' ','') for i in x])
  df_new['overview'] = df_new['overview'].apply(lambda x : [i.replace(' ','') for i in x])
  df_new['cast'] = df_new['cast'].apply(lambda x : [i.replace(' ','') for i in x])
  df_new['crew'] = df_new['crew'].apply(lambda x : [i.replace(' ','') for i in x])
  df_new['keywords'] = df_new['keywords'].apply(lambda x : [i.replace(' ','') for i in x])
  df_new['tags'] = df_new['overview'] + df_new['genres'] + df_new['keywords'] + df_new['cast'] + df_new['crew']
  new_df = df_new[['movie_id','title','tags']]
  new_df['tags'] = new_df['tags'].apply(lambda x: ' '.join(x))
  new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
  vectors = preprocess(new_df)
  simlarity = model(vectors)
  recommend(simlarity,new_df ,'Batman Begins')
  save_model(simlarity, new_df)
  

if __name__ == '__main__':
  main()

