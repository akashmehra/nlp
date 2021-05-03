
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from collections import Counter
import os
from tqdm import tqdm

class ToTensor():
    def __call__(self, sample):
        sample['source'] = torch.LongTensor(sample['source'])
        sample['target'] = torch.LongTensor(sample['target'])
        return sample


class ToNumpy():
    def __call__(self, sample):
        sample['source'] = sample['source'].numpy()
        sample['target'] = sample['target'].numpy()
        return sample

class Dictionary(object):
    
    def __init__(self):
        
        self.dont_care = 0
        self.bos_token = 1
        self.eos_token = 2
        self.unk_token = 3
        self.bos_word = '<bos>'
        self.eos_word = '<eos>'
        self.unk_word = '<unk>'
        
        self._word2idx = {self.bos_word: self.bos_token, 
                          self.eos_word: self.eos_token, 
                          self.unk_word: self.unk_token}
        
        self._idx2word = [self.dont_care, self.bos_word, self.eos_word, self.unk_word]
        self._dist = Counter()
        
    def add_word(self, word):
        self._dist[word] += 1
        if word not in self._word2idx:
            self._idx2word.append(word)
            self._word2idx[word] = len(self._idx2word) - 1
        #assert len(self._dist) == len(self._idx2word) == len(self._word2idx)
        return self._word2idx[word]

    def idx_to_word(self, idx):
        return self._idx2word[idx]
    
    def word_to_idx(self, word):
        return self._word2idx[word]
    
    def __add__(self, other):
        pass
        
    def __iadd__(self, other):
        widmap = other.wordidmap
        for k,v in widmap.items():
            if k not in self._word2idx:
                self._word2idx[k] = len(self._word2idx)
                self._idx2word.append(k)
        return self
    
    @property
    def wordidmap(self):
        return self._word2idx
    
    @property
    def idxwordmap(self):
        return self._idx2word
    
    @property
    def word_count(self):
        return self._dist
        

    def __len__(self):
        return len(self._idx2word)


class SentenceDataset(Dataset):
    def __init__(self, path, transform=None, dictionary=None):
        self._dictionary = dictionary
        self._tokens = self._tokenize(path)
        self.transform = transform
    
    def _tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        doc = []
        tokens = []
        self._num_tokens = 0
        # read lines from file.
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                if len(line.strip()) == 0:
                    continue
                words = [word.lower() for word in line.strip().split()]
                self._num_tokens += len(words)
                doc.append(words)
                
        for words in tqdm(doc): 
            tokens.append([self._dictionary.word_to_idx(word) for word in words])
                
        return tokens
    
    def __len__(self):
        return len(self._tokens)
    
    @property
    def corpus_size(self):
        return self._num_tokens
    
    def numpy(self):
        samples = []
        transform = ToNumpy()
        for idx in range(len(self._tokens)):
            sample = self[idx]
            sample = transform(sample)
            samples.append(sample)
        return samples
    
    @property
    def vocab(self):
        return self._dictionary
        
    def sentence(self, idx):
        return ' '.join([self._dictionary.idx_to_word(token) for token in self._tokens[idx]])
    
    def __getitem__(self, idx):
        tokens = self._tokens[idx]
        sample = {
            'source': [self._dictionary.bos_token] + tokens,
            'target': tokens + [self._dictionary.eos_token]
        }
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def __len__(self):
        return len(self._tokens)

class Wiki2Dataset():
    def __init__(self, path, transform=ToTensor()):
        self.transform = transform
        
        paths = [os.path.join(path, 'train.txt'), 
                 os.path.join(path, 'valid.txt'), 
                 os.path.join(path, 'test.txt')]
        
        self.vocab = self._create_vocab(paths)
        
       
        
        self.train_dataset = SentenceDataset(os.path.join(path, 'train.txt'), 
                                             transform=transform, dictionary=self.vocab)
        self.valid_dataset = SentenceDataset(os.path.join(path, 'valid.txt'), 
                                             transform=transform, dictionary=self.vocab)
        self.test_dataset = SentenceDataset(os.path.join(path, 'test.txt'), 
                                            transform=transform, dictionary=self.vocab)
        
        print(f"| Vocab Size: {len(self.vocab): 4d}")
        
    def _create_vocab(self, paths):
        vocab = Dictionary()
        for path in paths:
            # Add words to the dictionary
            with open(path, 'r', encoding="utf8") as f:
                for line in f:
                    if len(line.strip()) == 0:
                        continue
                    words = [word.lower() for word in line.strip().split()]
                    for word in words:
                        vocab.add_word(word)
        return vocab




def seq_collate_fn(data):
    
    def padding(seqs, seq_lens, dtype=torch.LongTensor):
        batch_size = len(seqs)
        max_seq_len = max(seq_lens)
        source_padded = torch.zeros(batch_size, max_seq_len).type(dtype)
        target_padded = torch.zeros(batch_size, max_seq_len).type(dtype)
        for i, seq in enumerate(seqs):
            end = len(seq['source'])
            source_padded[i,:end] = seq['source'] 
            target_padded[i,:end] = seq['target']
        
        return source_padded, target_padded
        
    data.sort(key=lambda d: len(d['source']), reverse=True)
    seq_lens = [len(d['source']) for d in data]
    source_padded, target_padded = padding(data, seq_lens)
    samples = {
        'sources': source_padded,
        'targets': target_padded,
        'seq_lens': seq_lens,
    }
    
    return samples
    
    
