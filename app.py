import pandas as pd
import streamlit as st
import pickle
import requests


def fetch_poster(movie_id):
    response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=020b311fe0559698373a16008dc6a672&language=en-US'.format(movie_id))
    data = response.json()
    return "https://image.tmdb.org/t/p/w500/" + data['poster_path']
def recommend(movie):
    movie_index = new_df[new_df['title']== movie].index[0]
    distance = simlarity[movie_index]
    movie_list = sorted(list(enumerate(distance)),reverse= True,key = lambda x:x[1])[1:6]
    movie_recommend = []
    movie_recommend_poster = []
    for i in movie_list:
        movie_id = new_df.iloc[i[0]].movie_id
        movie_recommend.append(new_df.iloc[i[0]].title)
        movie_recommend_poster.append(fetch_poster(movie_id))
    return   movie_recommend, movie_recommend_poster
movies =  pickle.load(open('df.pkl','rb'))
new_df = pd.DataFrame(movies)
simlarity =  pickle.load(open('sim.pkl','rb'))

st.title('movie recommendation system')
select_movie =st.selectbox('which movie do you like',new_df['title'].values)

if st.button('recommend'):
    names , poster = recommend(select_movie)
    col1 , col2 ,col3 , col4 , col5 = st.columns(5)
    with col1:
        st.text(names[0])
        st.image(poster[0])
    with col2:
        st.text(names[1])
        st.image(poster[1])
    with col3:
        st.text(names[2])
        st.image(poster[2])
    with col4:
        st.text(names[3])
        st.image(poster[3])
    with col5:
        st.text(names[4])
        st.image(poster[4])



