import pandas as pd
from time import time
from nltk.tokenize import line_tokenize, word_tokenize
from tqdm import tqdm



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

max_length_sent = 0
for song_id, song_info in tqdm(data.iterrows()):
    doc = line_tokenize(song_info.lyric)
    index = max(range(len(doc)), key=lambda i: len(word_tokenize(doc[i])))
    curr_max_length = len(word_tokenize(doc[index]))
    max_length_sent = max(max_length_sent, curr_max_length)
    for sent in doc:
        lyrics_pos_tagger['artist'].append(song_info.artist)
        lyrics_pos_tagger['song_name'].append(song_info.song_name)
        lyrics_pos_tagger['song_id'].append(song_id)
        lyrics_pos_tagger['sent'].append(sent)

    
print('Time to extract POS tagging: {} mins'.format(round((time() - t) / 60, 2)))
df = pd.DataFrame(lyrics_pos_tagger)
df.to_csv('Sentences_15klyrics.csv', index=False)
print('saved!')