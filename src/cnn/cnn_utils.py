from collections import Counter
from torchtext.vocab import build_vocab_from_iterator, vocab


class LetterTokenizer:
    def __call__(self, items):
        if isinstance(items, str):
            return self.__tokenize_str(items)
        else:
            return (self.__tokenize_str(t) for t in items)

    def __tokenize_str(self, t):
        tokenized = list(t.replace("\n", ""))
        tokenized.append("<eos>")
        tokenized.insert(0, "<bos>")
        return tokenized


def build_vocab(dataset, tokenizer, use_padding):
    counter = Counter()
    for i in range(len(dataset)):
        counter.update(tokenizer(dataset[i][0]))
    #     print(counter.most_common())
    builded_voc = vocab(counter)
    if use_padding:
        builded_voc.append_token("<pad>")
    builded_voc.insert_token("<unk>", 0)
    builded_voc.set_default_index(0)
    return builded_voc