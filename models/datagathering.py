from tswift import *
import pandas as pd
from langdetect import detect
import requests
import heapq
import xmltodict
import numpy as np
# %matplotlib inline

SELECTED_GENRES = ['Hip Hop', 'Metal', 'Folk', 'Rock', 'Punk',
                   'Country', 'Electronic', 'Blues', 'Jazz', 'Pop', 'Reggae']
hip_hop_genres = ['rap', 'funk', 'hip', 'hop']
metal_genres = ['metalcore', 'metal']
folk_genres = ['banjo', 'folk']
rock_genres = ['beat', 'roll', 'indie', 'rock']
punk_genres = ['grindcore', 'emo', 'punk']
electronic_genres = ['techno', 'idm', 'house', 'edm',
                     'disco', 'synthpop', 'tropical', 'electronic']
blues_genres = ['r&b', 'soul', 'blues']
pop_genres = ['electropop', 'new wave', 'pop']
jazz_genres = ['jazz']
reggae_genres = ['reggae']
country_genres = ['country']
filter_cols = ['acousticness', 'danceability', 'energy', 'duration_ms',
               'instrumentalness', 'valence', 'tempo', 'liveness', 'loudness', 'speechiness']

data = pd.read_csv('data/spotify_dataset_kg/data_w_genres.csv')
data_song = pd.read_csv('data/spotify_dataset_kg/data.csv')
data_genres = pd.read_csv('data/spotify_dataset_kg/data_by_genres.csv')

data_clean = data[data['genres'].str.len() >= 3]


def string_to_list(text):
    text = text.replace("[", "").replace(
        "'", "").replace("'", "").replace("]", "")
    return text.split(", ")


def cosine_similarity(x, y):
    return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))


def check_genres(genres):
    for item in genres:
        for gr in SELECTED_GENRES:
            gr = gr.replace(" ", "").lower()
            item = item.replace(" ", "").lower()
            if item.__contains__(gr):
                return True
    return False


def normalized_genre(genre):
    for gr in SELECTED_GENRES:
        grs = gr.lower()
        itemk = genre.lower()
        if [gn for gn in hip_hop_genres if(gn in itemk)]:
            return 'Hip Hop'
        if [gn for gn in metal_genres if(gn in itemk)]:
            return 'Metal'
        if [gn for gn in folk_genres if(gn in itemk)]:
            return 'Folk'
        if [gn for gn in rock_genres if(gn in itemk)]:
            return 'Rock'
        if [gn for gn in punk_genres if(gn in itemk)]:
            return 'Punk'
        if [gn for gn in country_genres if(gn in itemk)]:
            return 'Country'
        if [gn for gn in electronic_genres if(gn in itemk)]:
            return 'Electronic'
        if [gn for gn in blues_genres if(gn in itemk)]:
            return 'Blues'
        if [gn for gn in jazz_genres if(gn in itemk)]:
            return 'Jazz'
        if [gn for gn in pop_genres if(gn in itemk)]:
            return 'Pop'
        if [gn for gn in reggae_genres if(gn in itemk)]:
            return 'Reggae'
    return False


def get_lyrics(artist, song):
    try:
        url = "http://api.chartlyrics.com/apiv1.asmx/SearchLyricDirect"
        querystring = {"artist": artist, "song": song}
        response = requests.request("GET", url, params=querystring)
        docs = xmltodict.parse(response.text)
        return docs['GetLyricResult']['Lyric']
    except Exception as e:
        return None


def check_is_english_song(lyric):
    return detect(lyric) == 'en'


cleaned_genre = data_genres.copy()
cleaned_genre['normalized_genre'] = 'Not Classified'
for index, row in cleaned_genre.iterrows():
    new_genre = normalized_genre(row.genres)
    if (new_genre):
        cleaned_genre.loc[cleaned_genre.genres ==
                          row.genres, "normalized_genre"] = new_genre
# cleaned_genre=cleaned_genre[cleaned_genre.normalized_genre.notnull()]
cleaned_genre = cleaned_genre.drop(columns=['key', 'mode', 'popularity'])

COUNT_DATA = 0
LIMITS = 10000

new_data = {
    'artist': [],
    'song_name': [],
    'closest_genre': [],
    'lyric': [],
}

for index, row in data_song[13595:].iterrows():
    if (COUNT_DATA % 100 == 0):
        print(f'COUNT DATA: {COUNT_DATA}')
    if COUNT_DATA >= LIMITS:
        break
    try:
        artist_name = row['artists'].replace(
            '[', '').replace(']', '').replace("'", '')
        artist_data = data_clean[data_clean.artists.str.contains(artist_name)]
        song_name = row['name']
        # print(artist_name, '-', song_name, '-',
        #       check_genres(artist_data.genres))
        lyric = get_lyrics(artist_name, song_name)
        if lyric == None:
            continue
        if not (check_is_english_song(lyric)):
            continue
        # if not (check_genres(artist_data.genres)):
        #     continue
        if len(artist_data.genres) < 1:
            continue
        temp_similarity_mat = {}
        temp_l = string_to_list(artist_data.genres.values[0])
        if len(cleaned_genre[cleaned_genre.genres.isin(temp_l)]) < 1:
            continue
        for _, rowk in cleaned_genre[cleaned_genre.genres.isin(temp_l)].iterrows():
            temp_similarity_mat[rowk.genres] = cosine_similarity(
                rowk[filter_cols].to_numpy(), artist_data[filter_cols].to_numpy()[0])
        top_genre = heapq.nlargest(1, temp_similarity_mat)
        closest_genre = cleaned_genre[cleaned_genre.genres.isin(
            top_genre)].normalized_genre.values[0]
        if closest_genre == 'Not Classified':
            continue
        new_data['artist'].append(artist_name)
        new_data['song_name'].append(song_name)
        new_data['lyric'].append(lyric)
        new_data['closest_genre'].append(closest_genre)
        COUNT_DATA += 1
    except Exception as e:
        continue

song_genres = pd.DataFrame(new_data)
song_genres.to_csv(f'song_genres_{COUNT_DATA}_{index}.csv')
print("Saved")
