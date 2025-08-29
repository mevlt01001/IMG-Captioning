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
                             dropout=(0.1 if num_layer>1 else 0.0)).to(device)
        self.head  = nn.Linear(hidden_size, vocab_size, bias=False).to(device)
        self.h0_fc = nn.Linear(D_img, hidden_size).to(device)
        self.c0_fc = nn.Linear(D_img, hidden_size).to(device)
        self.pad_id = pad_id 
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.L = num_layer

    def _init_state(self, img_seq:torch.Tensor):          # img_seq: [B,S,D]
        g = img_seq.mean(1)                  # [B,D]
        h0 = torch.tanh(self.h0_fc(g)).unsqueeze(0).repeat(self.L,1,1)  # [L,B,H]
        c0 = torch.tanh(self.c0_fc(g)).unsqueeze(0).repeat(self.L,1,1)  # [L,B,H]
        return h0, c0

    # TRAINING
    def forward(self, img_seq, tokens_in):   # [B,S,D], [B,T]
        assert tokens_in is not None, "tokens_in cannot be None"
        h0, c0 = self._init_state(img_seq)
        x = self.embed(tokens_in)            # [B,T,E]
        y, _ = self.lstm(x, (h0, c0))        # [B,T,H]
        logits = self.head(y)                # [B,T,V]
        return logits

    # INFERENCE
    @torch.no_grad()
    def generate(self, img_seq, max_len=50):
        h, c = self._init_state(img_seq)
        B = img_seq.size(0)
        tok = torch.full((B,1), self.bos_id, dtype=torch.long, device=img_seq.device) # start tokens
        out = []
        for _ in range(max_len):
            x = self.embed(tok[:, -1:])
            y, (h,c) = self.lstm(x, (h,c))
            nxt = self.head(y[:, -1, :]).argmax(-1, keepdim=True)
            out.append(nxt)
            tok = torch.cat([tok, nxt], 1)
            if (nxt == self.eos_id).all(): break
        return torch.cat(out, 1)