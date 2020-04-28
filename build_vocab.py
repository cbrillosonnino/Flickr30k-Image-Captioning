from tqdm import tqdm
import nltk
from collections import Counter
from time import time
import multiprocessing as mp
from utils import Vocabulary
import pickle
import os

def build_vocab(ann_file = '../flickr30k-captions/results_20130124.token', threshold = 3):
    """Build a simple vocabulary wrapper."""
    punc_set = set([',',';',':','.','?','!','(',')'])
    counter = Counter()
    caption_list = []
    split = pickle.load(open('train_set.p', 'rb'))
    ann_file = os.path.expanduser(ann_file)
    with open(ann_file) as fh:
        for line in fh:
            img, caption = line.strip().split('\t')
            if img[:-2] in split:
                caption_list.append(caption)

    pool = mp.Pool(mp.cpu_count())
    tokens = pool.map(nltk.tokenize.word_tokenize, [caption.lower() for caption in tqdm(caption_list)])
    pool.close()
    tokens = [ item for elem in tokens for item in elem]
    tokens = [elem for elem in tokens if elem not in punc_set]
    counter = Counter(tokens)

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')
    vocab.add_word('<break>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)

    pickle.dump(vocab, open('vocab.p', 'wb'))

if __name__ == '__main__': build_vocab()
