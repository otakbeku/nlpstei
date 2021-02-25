import pandas as pd
import heapq
from nltk.tokenize import wordpunct_tokenize
import nltk
import numpy as np

from nltk.corpus import stopwords
from time import time
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
import multiprocessing
from gensim.test.utils import get_tmpfile

# Load data
lyrics = pd.read_csv('lyrics_song_genres_15k.csv')
RANDOM_SEED = 101

# # Preprocessing
# lyrics_length_threshold = lyrics.lyric.apply(
#     lambda x: x.split()).apply(len).quantile(0.25)
# cleaned_lyrics = lyrics[(lyrics.length_lyric >= lyrics_length_threshold)]
# threshold = cleaned_lyrics.groupby('closest_genre')[
#     'song_name'].nunique()['Metal']
# cleaned_lyrics = cleaned_lyrics[cleaned_lyrics.closest_genre != 'Reggae']
# cleaned_lyrics = cleaned_lyrics.groupby('closest_genre').sample(
#     n=threshold, random_state=RANDOM_SEED)

cleaned_lyrics = lyrics.copy()


freq_unique_words_per_genre = {}
stop_words = stopwords.words('english')
sentences = []
for genre, lyric in cleaned_lyrics.groupby('closest_genre')['lyric'].sum().iteritems():
    text = lyric.lower()
    tokens = wordpunct_tokenize(text)
    # filtered_sentence = [w for w in tokens if (not w in stop_words) and (w.isalpha())]
    filtered_sentence = [w for w in tokens if (w.isalpha())]
    sentences += filtered_sentence
    fdist1 = nltk.FreqDist(filtered_sentence)
    freq_unique_words_per_genre[genre] = dict(
        (word, freq) for word, freq in fdist1.items() if word.isalpha())

sentences_1 = [sent.split() for sent in sentences]
common_terms = []
for key in freq_unique_words_per_genre.keys():
    common_terms += heapq.nlargest(
        2500, freq_unique_words_per_genre[key], key=freq_unique_words_per_genre[key].get)
common_terms = set(common_terms)

# Creating the parser
phrases = Phrases(sentences_1, common_terms=common_terms, min_count=10)
ngram = Phraser(phrases)
ngrams_sent = list(ngram[sentences_1])
# Adding some pseudo tag
# ngrams_sent.append(['paddingkosong'])
# ngrams_sent.append(['tidakdiketahui'])


# Train word2vec
path = get_tmpfile("fifteenklyricswv_w5ns5_notcleaned_nopad_2.model")
cores = multiprocessing.cpu_count()
wvmodel = Word2Vec(min_count=5,
                   window=5,
                   size=300,
                   sample=0.001,
                   alpha=0.03,
                   min_alpha=0.0007,
                   workers=cores-1)
t = time()
wvmodel.build_vocab(ngrams_sent, progress_per=1000)
print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

phrasest = time()
wvmodel.train(ngrams_sent, total_examples=wvmodel.corpus_count,
              epochs=50, report_delay=1)
print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
print(f'{np.shape(wvmodel.wv.vectors)}')
wvmodel.save('fifteenklyricswv_w5ns5_notcleaned_nopad_2.model')

print('saved!')
