class Tokenizer:
    def __init__(self, captions:list[str]):
        self.captions: list[str] = captions
        self.min_freq = 5
        self.BOS = "<BOS>"
        self.EOS = "<EOS>"
        self.PAD = "<PAD>"
        self.UNK = "<UNK>"
        self.build(captions) # captions, tokenized_captions

    def normalize(self, text:str) -> list[str]:
        keeped = []
        for word in text.split():
            word = word.lower()
            _w = ""
            for ch in word:
                if ch.isalpha():
                    _w += ch
            keeped.append(_w)
        return [self.BOS] + keeped + [self.EOS]

    def build(self, captions:list[str]):

        self.captions = []
        for sentence in captions:
            cap = self.normalize(sentence)
            self.captions.append(cap)
        
        freq = {}
        for cap in self.captions:
            for t in cap:
                freq[t] = freq.get(t, 0) + 1

        captions = []
        for cap in self.captions:
            sentence = []
            for w in cap:
                if freq[w] > self.min_freq:
                    sentence.append(w)
                else:
                    sentence.append(self.UNK)
            captions.append(sentence)
        
        self.captions = captions
        
        sepical_tokens = [self.PAD, self.UNK, self.BOS, self.EOS]
        self.vocap = list(set(sepical_tokens+[word for cap in self.captions for word in cap]))
        self.vocap_size = len(self.vocap)
        
        self.seq_len = max(len(cap) for cap in self.captions)
        print(f"seq_len: {self.seq_len}")
        self.captions = [cap + [self.PAD] * (self.seq_len - len(cap)) for cap in self.captions]
        
        self.char2idx = {t:i for i,t in enumerate(self.vocap)}
        self.idx2char = {i:t for t,i in self.char2idx.items()}

        captions = []
        for cap in self.captions:
            sentence = []
            for w in cap:
                sentence.append(self.char2idx[w])
            captions.append(sentence)
        self.tokenized_captions = captions

    def set_vocab(self, vocap:list[str]):
        self.vocap = vocap
        self.char2idx = {t:i for i,t in enumerate(self.vocap)}
        self.idx2char = {i:t for t,i in self.char2idx.items()}
        
        captions = []
        for cap in self.captions:
            sentence = []
            for w in cap:
                sentence.append(self.char2idx[w])
            captions.append(sentence)
        self.tokenized_captions = captions
        return self

    def encode(self, text:str) -> list[int]:
        toks = self.normalize(text)
        unk_id = self.char2idx[self.UNK]
        return [self.char2idx.get(t, unk_id) for t in toks]

    def decode(self, ids:list[int]) -> str:
        s = []
        pad = self.char2idx.get(self.PAD)
        bos = self.char2idx.get(self.BOS)
        eos = self.char2idx.get(self.EOS)
        for i in ids:
            if pad is not None and i == pad:   continue
            if bos is not None and i == bos:   continue
            if eos is not None and i == eos:   break
            s.append(self.idx2char.get(i, self.UNK))
        return " ".join(s)
