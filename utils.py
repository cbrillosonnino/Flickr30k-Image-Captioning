from torchvision.datasets import VisionDataset
import os
from collections import defaultdict
from PIL import Image
import numpy as np
import nltk
from collections import Counter
import pickle
import torch


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.all_words = []

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
            self.all_words.append(word)

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def get_id(self, w):
        return self.word2idx[w]

    def encode_seq(self, l):
        return [self.word2idx[i] if i in self.word2idx else self.word2idx['<unk>'] for i in l]

    def get_token(self, idx):
        return self.idx2word[idx]

    def decode_seq(self, l):
        return [self.idx2word[i] for i in l]


# Adapted from https://pytorch.org/docs/stable/_modules/torchvision/datasets/flickr.html#Flickr30k
class Custom_Flickr30k(VisionDataset):
    """`Flickr30k Entities <http://web.engr.illinois.edu/~bplumme2/Flickr30kEntities/>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        ann_file (string): Path to annotation file.
        vocabulary (object): Vocabulary wrapper
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        is_train (boolean): Train set or val set
    """

    def __init__(self, root, ann_file, vocabulary, train=True, transform=None, target_transform=None):
        super(Custom_Flickr30k, self).__init__(root, transform=transform, target_transform=target_transform)
        self.ann_file = os.path.expanduser(ann_file)
        self.train = train
        self.annotations = defaultdict(list)
        self.punc_set = set([',',';',':','.','?','!','(',')'])
        self.vocabulary = vocabulary

        if self.train:
            split = pickle.load(open('train_set.p', 'rb'))
            idx=0
            with open(self.ann_file) as fh:
                for line in fh:
                    img_id, caption = line.strip().split('\t')
                    img = img_id[:-2]
                    if img in split:
                        self.annotations[idx].extend([img,caption])
                        idx += 1
            self.ids = np.arange(idx-1)

        else:
            split = pickle.load(open('val_set.p', 'rb'))
            idx = -1
            last_image = ''
            with open(self.ann_file) as fh:
                for line in fh:
                    img_id, caption = line.strip().split('\t')
                    img = img_id[:-2]
                    if img in split:
                        if img != last_image:
                            idx += 1
                            last_image = img
                            self.annotations[idx].append(img)
                            self.annotations[idx].append([caption])
                        else:
                            self.annotations[idx][1].append(caption)
            self.ids = np.arange(idx)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        instance = self.annotations[index]

        img_id = instance[0]
        caption = instance[1]

        # Image
        filename = os.path.join(self.root, img_id)
        img = Image.open(filename).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.train:

            # Captions
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            caption = []
            caption.append(self.vocabulary('<start>'))
            caption.extend([self.vocabulary(token) for token in tokens if token not in self.punc_set])
            caption.append(self.vocabulary('<end>'))
            target = torch.Tensor(caption)

        else:
            # Captions
            target = []
            for item in caption:
                tokens = nltk.tokenize.word_tokenize(str(item).lower())
                target.extend([self.vocabulary(token) for token in tokens if token not in self.punc_set])
                target.append(self.vocabulary('<break>'))

            target = torch.Tensor(target[:-1])

        return img, target


    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, torch.tensor(lengths)
