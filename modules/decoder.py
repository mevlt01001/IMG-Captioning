import torch
import torch.nn as nn

class AdditiveAttn(nn.Module):
    def __init__(self, fdim=512, hdim=512, dim=256):
        super().__init__()
        self.linear_image = nn.Linear(fdim, dim, bias=False)
        self.linear_hidden = nn.Linear(hdim, dim, bias=False)
        self.linear_score = nn.Linear(dim, 1, bias=False)

    def forward(self, image_features, hidden_state):
        # image_features: [B, S, D_img], hidden_state:   [B, D_hid]

        proj_image  = self.linear_image(image_features)              # [B, S, A]
        proj_hidden = self.linear_hidden(hidden_state).unsqueeze(1)  # [B, 1, A]

        combined = torch.tanh(proj_image + proj_hidden)              # [B, S, A]
        scores = self.linear_score(combined).squeeze(-1)           # [B, S]
        weights = torch.softmax(scores, dim=-1)                     # [B, S]

        context = (weights.unsqueeze(-1) * image_features).sum(1)   # [B, D_img]
        return context, weights


class LSTMDecoder(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 token_feats, 
                 hidden_size, 
                 D_img, 
                 pad_id, 
                 bos_id, 
                 eos_id, 
                 device=torch.device('cpu')):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, token_feats, padding_idx=pad_id).to(device)
        self.head1 = nn.Linear(token_feats + D_img, token_feats, bias=False).to(device)
        self.cell  = nn.LSTMCell(token_feats, hidden_size).to(device)  # concat(emb, ctx)
        self.head  = nn.Linear(hidden_size, vocab_size, bias=False).to(device)
        self.attn  = AdditiveAttn(D_img, hidden_size, 256).to(device)
        self.h0_fc = nn.Linear(D_img, hidden_size).to(device)
        self.c0_fc = nn.Linear(D_img, hidden_size).to(device)
        self.bos_id, self.eos_id = bos_id, eos_id

    def _init_state(self, img_seq):  # [B,S,D]
        g  = img_seq.mean(1)                       # [B,D]
        h0 = torch.tanh(self.h0_fc(g))             # [B,H]
        c0 = torch.tanh(self.c0_fc(g))             # [B,H]
        return h0, c0

    # Teacher forcing
    def forward(self, img_seq, tokens_in):         # tokens_in:[B,T]
        B,T = tokens_in.shape
        h,c = self._init_state(img_seq)
        outs = []
        for t in range(T):
            emb  = self.embed(tokens_in[:, t])     # [B,E]
            ctx,_= self.attn(img_seq, h)           # [B,D]
            x    = torch.cat([emb, ctx], -1)       # [B, E+D]
            x    = self.head1(x)                   # [B, E]
            h,c  = self.cell(x, (h,c))             # [B,H]
            outs.append(self.head(h))              # [B,vocab]
        return torch.stack(outs, 1)                # [B,T,vocab]

    @torch.no_grad()
    def generate(self, img_seq, max_len=50):
        B     = img_seq.size(0)
        h,c   = self._init_state(img_seq)
        tok   = torch.full((B,), self.bos_id, device=img_seq.device, dtype=torch.long)
        outs  = []
        for _ in range(max_len):
            emb  = self.embed(tok)                 # [B,E]
            ctx,_= self.attn(img_seq, h)           # [B,D]
            x    = torch.cat([emb, ctx], -1)       # [B,E+D]
            x    = self.head1(x)                   # [B,E]
            h,c  = self.cell(x, (h,c))
            nxt  = self.head(h).argmax(-1)         # [B]
            outs.append(nxt)
            tok = nxt
            if (nxt == self.eos_id).all(): break
        return torch.stack(outs, 1)                # [B,T]
