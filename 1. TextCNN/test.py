from gensim.models import KeyedVectors
from gensim.models import Word2Vec 

model = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)
model.build_vocab()