# run this first
import stanza as st
st.download('en')

import gensim.downloader as api
corpus = api.load('glove-wiki-gigaword-300', return_path=True)