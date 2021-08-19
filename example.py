import logging
import random
from collections import OrderedDict
from random import choice

import torch
from torch.utils.data import Dataset

from mingpt.model import GPTConfig, GPT
from mingpt.trainer_acc import Trainer, TrainerConfig
from mingpt.utils import sample


class Tokenizer:
    def __init__(self, data, vocab_size):
        self.vocab_size = vocab_size
        self.vocab = self.build_vocab(data)

        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}

    def sort_vocab(self, vocab):
        """
        Vocab should have the followind order: hashtag, numbers, characters sorted by length.
        Hashtags should go first, because they will be used as dividers on tokenization step.
        Numbers should go before characters, because token ids are numbers. Otherwise token ids will be considered as usual numbers and replaced twice.
        """
        sorted_vocab = sorted(vocab, key=lambda x: len(x), reverse=True)
        tag = [int(s) for s in sorted_vocab if s == '#']

        numeric = [int(s) for s in sorted_vocab if s.isnumeric()]
        numeric_str = [str(s) for s in sorted(numeric, reverse=True)]
        rest = [s for s in sorted_vocab if not s.isnumeric()]

        sorted_vocab = tag + numeric_str + rest

        return sorted_vocab

    def build_vocab(self, data):
        logging.info("... build vocab")

        """
        Build vocabluary using BPE alghorithm.
        """
        vocab = set(data)
        if len(vocab) > self.vocab_size:
            raise ValueError('Vocab size should be greater than unique char count')

        # check all available characters
        char_set = {c for c in vocab if c.isalpha()}

        # candidates dictionary will contain a set of all available tokens to search
        candidate_dict = dict().fromkeys(char_set, 0)

        # occurrences will contain all matched tokens and the count, how many times the token has been found.
        token_occurrences = OrderedDict()
        while len(vocab) < self.vocab_size:
            for candidate in candidate_dict.keys():
                occurrences = data.count(candidate)
                candidate_dict[candidate] = occurrences

            candidate_dict = {candidate: count for candidate, count in candidate_dict.items() if count}
            vocab.update(set(candidate_dict.keys()))
            token_occurrences.update(candidate_dict)

            # build new candidates
            temp_candidate_set = set()
            for char in char_set:
                # don't test candidates with occurency <= 2. New candidates won't have occurency higher than 2
                temp_candidate_set.update(
                    {candidate + char for candidate in candidate_dict.keys() if token_occurrences[candidate] > 2})

            candidate_dict = dict().fromkeys(temp_candidate_set, 0)

        tokens_to_remove = len(vocab) - self.vocab_size
        token_occurrences = OrderedDict(sorted(token_occurrences.items(), key=lambda x: x[1], reverse=True))
        for _ in range(tokens_to_remove):
            token, _ = token_occurrences.popitem()
            vocab.remove(token)

        sorted_vocab = self.sort_vocab(vocab)

        # add a special token for unknown tokens
        sorted_vocab.append('<unk>')
        self.vocab_size += 1  # plus <unk> special token

        logging.info("... done: {}".format(len(sorted_vocab)))
        return sorted_vocab

    def tokenize(self, data, block_size):
        for token in self.vocab:
            data = data.replace(token, f'#{self.stoi[token]}#')

        # If everything went well, first and last characters won't have # pair. Need to trim them
        data = data[1:-1]
        # Split by ## pairs
        tokenized_text = data.split('##')
        # Filter empty strings
        tokenized_text = [x for x in tokenized_text if x]
        result = []
        for tokenized in tokenized_text:
            # In case other single # found, replace them with <unk> special token, marking the element as unknown
            if '#' in tokenized:
                for unknown_candidate in tokenized.split('#'):
                    if unknown_candidate.isnumeric():
                        result.append(self.itos[int(unknown_candidate)])
                    else:
                        result.append('<unk>')
            else:
                result.append(self.itos[int(tokenized)])

        # all texts should have equal size. We can make text length equal by filling text with spaces
        for _ in range(block_size - len(result)):
            result.append(' ')

        # in case the sentence is longer, than block_size, we trim the sentence
        return result[:block_size]

    def encode(self, data):
        return [self.stoi[s] for s in data]

    def decode(self, data):
        return ''.join([self.itos[int(i)] for i in data])


class WordDataset(Dataset):

    def __init__(self, original, modern, tokenizer_original, tokenizer_modern, block_size):
        self.tokenizer_original = tokenizer_original
        self.tokenizer_modern = tokenizer_modern

        self.block_size = block_size * 2
        self.original = [tokenizer_original.tokenize(t, block_size) for t in original]
        self.modern = [tokenizer_modern.tokenize(t, block_size) for t in modern]

    def __len__(self):
        return len(self.original)

    def __getitem__(self, idx):
        """
        The idea is to get a sentence in a modern English
        and translate it to Shakespeare English.

        In the init method we already split a sentence into tokens and filled with spaces,
        to have an equal sentence size. In this method we just encode the tokens to
        ids (a list of numbers), and we're trying to map ids sequences
        (original Englisn and modern English)
        """

        modern_text = self.tokenizer_modern.encode(self.modern[idx])
        original_text = self.tokenizer_original.encode(self.original[idx])
        dix = modern_text + original_text

        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        y[:int(self.block_size / 2) - 1] = -100

        return x, y


def _main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    vocab_size = 10000
    text_modern = open('all_modern.txt', 'r', encoding='utf-8').read()
    text_original = open('all_original.txt', 'r', encoding='utf-8').read()

    tokenizer_modern = Tokenizer(text_modern, vocab_size)
    tokenizer_original = Tokenizer(text_original, vocab_size)

    texts = list(zip(text_modern.splitlines(), text_original.splitlines()))
    random.shuffle(texts)
    text_modern, text_original = zip(*texts)

    train_dataset_size = round(0.85 * len(text_modern))
    test_dataset_size = round(0.1 * len(text_modern))
    valid_dataset_size = round(0.05 * len(text_modern))

    train_modern = text_modern[:train_dataset_size]
    test_modern = text_modern[train_dataset_size:train_dataset_size + test_dataset_size]
    valid_modern = text_modern[-valid_dataset_size:]

    train_original = text_original[:train_dataset_size]
    test_original = text_original[train_dataset_size:train_dataset_size + test_dataset_size]
    valid_original = text_original[-valid_dataset_size:]

    block_size = 100  # the estimate how long lines the text could be (token count)

    train_dataset = WordDataset(train_original, train_modern, tokenizer_original, tokenizer_modern, block_size)
    test_dataset = WordDataset(test_original, test_modern, tokenizer_original, tokenizer_modern, block_size)

    mconf = GPTConfig(tokenizer_original.vocab_size, train_dataset.block_size,
                      n_layer=2, n_head=4, n_embd=512)
    model = GPT(mconf)

    tokens_per_epoch = len(train_dataset) * block_size
    train_epochs = 20

    # initialize a trainer instance and kick off training
    tconf = TrainerConfig(max_epochs=train_epochs, batch_size=64, learning_rate=3e-4,
                          lr_decay=True, warmup_tokens=tokens_per_epoch, final_tokens=train_epochs * tokens_per_epoch,
                          num_workers=2)
    trainer = Trainer(model, train_dataset, test_dataset, tconf)
    trainer.train()

    for _ in range(5):
        idx = choice(range(len(valid_original)))

        context = valid_modern[idx]
        x = torch.tensor(tokenizer_modern.encode(tokenizer_modern.tokenize(context, block_size)), dtype=torch.long)[
            None, ...].to(trainer.device)
        y = sample(model, x, block_size, temperature=1.0, sample=True, top_k=10)[0]

        predicted = y[block_size:]
        completion = tokenizer_original.decode(predicted)
        print(f'Modern:             {context}')
        print(f'Predicted original: {completion}')
        print(f'Real original:      {valid_original[idx]}')
        print('--------------------------------------------------')


if __name__ == '__main__':
    _main()
