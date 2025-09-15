import math
import torch
import torch.nn as nn

@torch.no_grad()
def sinusoidal_1d_pe(S: int, D: int, device=None) -> torch.Tensor:

    assert D % 2 == 0, f"D % 2 == 0 must; D={D}"
    device = device or torch.device('cpu')

    d = D // 2
    pos = torch.arange(S, device=device, dtype=torch.float32)               # [S]
    k   = torch.arange(d, device=device, dtype=torch.float32)               # [d]
    omega = torch.exp(-math.log(10000.0) * k / d)                           # [d]

    # Broadcast: [S,1]*[d] -> [S,d]
    pe_sin = torch.sin(pos[..., None] * omega)  # [S,d]
    pe_cos = torch.cos(pos[..., None] * omega)  # [S,d]

    pe = torch.cat([pe_sin, pe_cos], dim=-1).unsqueeze(0).contiguous()  # [1,S,D]
    return pe

def causal_mask(T:int, device):
    """
    ```
    tensor([[False,  True,  True,  True],
        [False, False,  True,  True],
        [False, False, False,  True],
        [False, False, False, False]])
    ```
    """
    m = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
    return m  # [T,T]


class DecoderAttnBlock(nn.Module):
    def __init__(self, 
                 embed_dim:int, 
                 heads:int=8, 
                 dropout:float=0.1, 
                 mlp_ratio:float=4.0):
        
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, heads, batch_first=True, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(embed_dim, heads, batch_first=True, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim*mlp_ratio)), 
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim*mlp_ratio), embed_dim), 
            nn.Dropout(dropout)
        )

    def forward(self, 
                x:torch.Tensor, 
                enc_out, 
                self_attn_mask=None, 
                self_key_padding=None, 
                enc_key_padding=None):
        
        # x: [B,T,D], enc_out: [B,S,D]
        h = self.norm1.forward(x)
        x = x + self.self_attn.forward(h, h, h,
                               attn_mask=self_attn_mask,            # [T,T] mask
                               key_padding_mask=self_key_padding,   # [B,T]
                               need_weights=False)[0]

        h = self.norm2.forward(x)
        x = x + self.cross_attn.forward(h, enc_out, enc_out,
                                key_padding_mask=enc_key_padding,   # [B,S]
                                need_weights=False)[0]
        h = self.norm3.forward(x)
        x = x + self.mlp.forward(h)
        return x  # [B,T,D]
    
class ViTDecoder(nn.Module):
    def __init__(self, 
                 vocab_size:int, 
                 pad_id:int,
                 bos_id:int,
                 eos_id:int,
                 dim:int, 
                 depth:int=4, 
                 heads:int=8, 
                 dropout:float=0.1,
                 device=torch.device('cpu')):
        
        super().__init__()
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.embed  = nn.Embedding(vocab_size, dim, padding_idx=pad_id).to(device)
        self.dropout = nn.Dropout(dropout).to(device)
        self.blocks = nn.ModuleList([DecoderAttnBlock(dim, heads, dropout) for _ in range(depth)]).to(device)
        self.norm = nn.LayerNorm(dim).to(device)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False).to(device)
        # self.lm_head.weight = self.embed.weight

    def forward(self, 
                enc_out:torch.Tensor, # [B,S,D]
                tins:torch.Tensor):   # [B,T]

        B, T = tins.shape
        x = self.embed.forward(tins)                           # [B,T,D]
        x = x + sinusoidal_1d_pe(T, x.size(-1), x.device)  # 1D PE
        x = self.dropout.forward(x)

        attn_mask = causal_mask(T, x.device)               # [T,T]
        key_pad   = (tins == self.pad_id)                  # [B,T]
        enc_key_pad = None                                 

        for blk in self.blocks:
            x = blk.forward(x, enc_out, self_attn_mask=attn_mask, self_key_padding=key_pad, enc_key_padding=enc_key_pad)

        x = self.norm(x)
        logits = self.lm_head(x)                        # [B,T,vocab]
        return logits

    @torch.no_grad()
    def generate(self, 
                 enc_out,
                 max_len:int=256
                 ):
        B, S, D = enc_out.shape
        x_ids = torch.full((B,1), self.bos_id, device=enc_out.device, dtype=torch.long)
        finished = torch.zeros(B, dtype=torch.bool, device=enc_out.device)

        for t in range(1, max_len+1):
            x = self.embed(x_ids)                                  # [B,t,D]
            x = x + sinusoidal_1d_pe(x.size(1), x.size(2), x.device)
            for blk in self.blocks:
                attn_mask = causal_mask(x.size(1), x.device)
                key_pad   = (x_ids == self.pad_id)
                x = blk(x, enc_out, self_attn_mask=attn_mask, self_key_padding=key_pad)

            x = self.norm(x)
            logits = self.lm_head(x[:, -1])                        # [B,vocab]
            next_id = torch.argmax(logits, dim=-1)                 # greedy
            x_ids = torch.cat([x_ids, next_id[:,None]], dim=1)

            finished = finished | (next_id == self.eos_id)
            if bool(finished.all()):
                break

        return x_ids  # [B, <=max_len+1]

# ------ OLD VERSION ------
class AdditiveAttn(nn.Module):
    def __init__(self, fdim=512, hdim=512, dim=128):
        super().__init__()

        self.linear_image  = nn.Linear(fdim, dim, bias=False)
        self.linear_hidden = nn.Linear(hdim, dim, bias=False)
        self.linear_score  = nn.Linear(dim, 1,  bias=False)

    @torch.no_grad()
    def precompute_image(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        image_features: [B, S, D_img]
        return: proj_image [B, S, A]  (A=dim)
        """
        return self.linear_image(image_features)

    def forward_cached(self,
                       proj_image: torch.Tensor,      # [B, S, A] precomputed
                       image_features: torch.Tensor,  # [B, S, D_img] bmm
                       hidden_state: torch.Tensor     # [B, H]
                       ) -> tuple[torch.Tensor, torch.Tensor]:
        
        proj_hidden = self.linear_hidden(hidden_state).unsqueeze(1)
        combined = torch.tanh(proj_image + proj_hidden)
        scores = self.linear_score(combined).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)

        # [B,1,S] x [B,S,D] -> [B,1,D] -> [B,D]
        context = torch.bmm(weights.unsqueeze(1), image_features).squeeze(1)
        return context, weights

class LSTMDecoder(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 token_feats, 
                 hidden_size, 
                 D_img, 
                 pad_id, bos_id, eos_id, 
                 device=torch.device('cpu')):
        super().__init__()
        self.device = device

        self.embed = nn.Embedding(vocab_size, token_feats, padding_idx=pad_id)
        self.ctx_proj = nn.Linear(D_img, token_feats, bias=False)
        self.mix = nn.Sequential(
            nn.Linear(token_feats*2, token_feats, bias=False),
            nn.ReLU(inplace=True),
            nn.LayerNorm(token_feats)
        )
        self.cell  = nn.LSTMCell(token_feats, hidden_size)
        self.head  = nn.Linear(hidden_size, vocab_size, bias=False)
        self.attn  = AdditiveAttn(D_img, hidden_size, dim=128)

        self.h0_fc = nn.Linear(D_img, hidden_size)
        self.c0_fc = nn.Linear(D_img, hidden_size)

        self.bos_id, self.eos_id = bos_id, eos_id
        self.to(device)

    def _init_state(self, img_seq):  # [B,S,D]
        g  = img_seq.mean(1)                     # [B,D]
        h0 = torch.tanh(self.h0_fc(g)) * 0.5
        c0 = torch.tanh(self.c0_fc(g)) * 0.5
        return h0, c0

    def forward(self, img_seq, tokens_in):  # tokens_in:[B,T]
        B, T = tokens_in.shape
        img_seq = img_seq.contiguous()

        proj_image = self.attn.precompute_image(img_seq)  # [B,S,A]
        h, c = self._init_state(img_seq)

        V  = self.head.out_features
        outs = torch.empty(B, T, V, device=img_seq.device, dtype=torch.float32)

        for t in range(T):
            emb = self.embed(tokens_in[:, t])            # [B,E]
            ctx, _ = self.attn.forward_cached(proj_image, img_seq, h)  # [B,D]
            ctxE = self.ctx_proj(ctx)                    # [B,E]
            x = torch.cat([emb, ctxE], dim=-1)           # [B,2E]
            x = self.mix(x) + emb                        # residual ile emb korunur
            h, c = self.cell(x, (h, c))                  # [B,H]
            outs[:, t, :] = self.head(h)                 # [B,V]
        return outs

    @torch.no_grad()
    def generate(self, img_seq, max_len=50):
        B = img_seq.size(0)
        img_seq = img_seq.contiguous()

        proj_image = self.attn.precompute_image(img_seq)
        h, c = self._init_state(img_seq)
        tok  = torch.full((B,), self.bos_id, device=img_seq.device, dtype=torch.long)

        V = self.head.out_features
        outs = torch.empty(B, max_len, dtype=torch.long, device=img_seq.device)

        for t in range(max_len):
            emb = self.embed(tok)                        # [B,E]
            ctx, _ = self.attn.forward_cached(proj_image, img_seq, h)
            ctxE = self.ctx_proj(ctx)                    # [B,E]
            x = torch.cat([emb, ctxE], dim=-1)           # [B,2E]
            x = self.mix(x) + emb
            h, c = self.cell(x, (h, c))
            nxt = self.head(h).argmax(-1)                # [B]
            outs[:, t] = nxt
            tok = nxt
            if (nxt == self.eos_id).all():
                return outs[:, :t+1]
        return outs
