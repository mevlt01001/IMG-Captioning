import torch
import torch.nn as nn

class LSTMDecoder(nn.Module):
    def __init__(self, 
                 vocab_size, # size of vocabulary
                 token_feats, # size of token features 
                 hidden_size, # size of hidden state
                 num_layer, # number of layers
                 D_img, # size of image token features
                 pad_id, # padding id
                 bos_id, # bos id
                 eos_id, # eos id
                 device = torch.device('cpu')
                 ):
        
        super().__init__()
        assert all([pad_id, bos_id, eos_id]) is not None, "pad_id, bos_id, eos_id cannot be None"
        self.embed = nn.Embedding(vocab_size, token_feats, padding_idx=pad_id).to(device)
        self.lstm  = nn.LSTM(token_feats, hidden_size, num_layers=num_layer, batch_first=True,
                             dropout=(0.25 if num_layer>1 else 0.0)).to(device)
        self.head  = nn.Linear(hidden_size, vocab_size, bias=False).to(device)
        self.img2tok = nn.Linear(D_img, token_feats).to(device)
        self.h0_fc = nn.Linear(D_img, hidden_size).to(device)
        self.c0_fc = nn.Linear(D_img, hidden_size).to(device)
        self.pad_id = pad_id 
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.L = num_layer

    def _init_state(self, img_seq:torch.Tensor): # img_seq: [B,S,D]
        g = img_seq.mean(1) # [B,D]
        h0 = torch.tanh(self.h0_fc(g)).unsqueeze(0).repeat(self.L,1,1)  # [L,B,H]
        c0 = torch.tanh(self.c0_fc(g)).unsqueeze(0).repeat(self.L,1,1)  # [L,B,H]
        v = self.img2tok(g).unsqueeze(1) # [B,1,E]
        return h0, c0, v

    # TRAINING
    def forward(self, img_seq, tokens_in):  # [B,S,D], [B,T]
        assert tokens_in is not None
        h0, c0, v = self._init_state(img_seq)
        x = self.embed(tokens_in) + v  # [B,T,E] 
        y, _ = self.lstm(x, (h0, c0))
        logits = self.head(y)
        return logits

    # INFERENCE
    @torch.no_grad()
    def generate(self, img_seq, max_len=50):
        h, c ,v = self._init_state(img_seq)
        B = img_seq.size(0)
        tok = torch.full((B,1), self.bos_id, dtype=torch.long, device=img_seq.device)
        out = []
        for _ in range(max_len):
            x = self.embed(tok[:, -1:]) + v
            y, (h,c) = self.lstm(x, (h,c))
            nxt = self.head(y[:, -1, :]).argmax(-1, keepdim=True)
            out.append(nxt)
            tok = torch.cat([tok, nxt], 1)
            if (nxt == self.eos_id).all():
                break
        return torch.cat(out, 1)
