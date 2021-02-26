import pandas as pd
from time import time
from nltk.tokenize import line_tokenize, word_tokenize
from tqdm import tqdm
import numpy as np



# stanza
import stanza as st


data = pd.read_csv('lyrics_song_genres_15k.csv')
# pos_tagger =  st.Pipeline(lang='en')

lyrics_pos_tagger = {
    'artist':[],
    'song_name':[],
    'song_id':[],
    'sent':[]
    # 'word':[],
    # 'XPOS':[],
    # 'UPOS':[],
}
# t = time()

# max_length_sent = 0
# for song_id, song_info in data.iterrows():
#     lines = '. '.join(line_tokenize(song_info.lyric))
#     doc = pos_tagger(lines)
#     for sent in doc.sentences:
#         curr_max_length = 0
#         for word in sent.words:
#             curr_max_length += 1
#             lyrics_pos_tagger['artist'].append(song_info.artist)
#             lyrics_pos_tagger['song_name'].append(song_info.song_name)
#             lyrics_pos_tagger['song_id'].append(song_id)
#             lyrics_pos_tagger['word'].append(word.text)
#             lyrics_pos_tagger['XPOS'].append(word.xpos)
#             lyrics_pos_tagger['UPOS'].append(word.upos)
#         max_length_sent = max(max_length_sent, curr_max_length)
t = time()
# max_length_sent = 20
for song_id, song_info in tqdm(data.iterrows()):
    doc = line_tokenize(song_info.lyric.replace(',','\n'))
    for sent in doc:
        # if len(word_tokenize(sent)) > max_length_sent:
        #     docx = np.array_split(word_tokenize(sent), max_length_sent)
        #     for sentx in docx:
        #         sentx = ' '.join(sentx)
        #         lyrics_pos_tagger['artist'].append(song_info.artist)
        #         lyrics_pos_tagger['song_name'].append(song_info.song_name)
        #         lyrics_pos_tagger['song_id'].append(song_id)
        #         lyrics_pos_tagger['sent'].append(sentx)
        #     continue
        lyrics_pos_tagger['artist'].append(song_info.artist)
        lyrics_pos_tagger['song_name'].append(song_info.song_name)
        lyrics_pos_tagger['song_id'].append(song_id)
        lyrics_pos_tagger['sent'].append(sent)

    
print('Time to extract POS tagging: {} mins'.format(round((time() - t) / 60, 2)))
df = pd.DataFrame(lyrics_pos_tagger)
# df.to_csv('sentences_15klyrics_mls_{}.csv'.format(max_length_sent), index=False)
df.to_csv('sentences_15klyrics.csv', index=False)
print('saved!')