import torch
import torch.nn as nn

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
                       proj_image: torch.Tensor,      # [B, S, A] -> precomputed
                       image_features: torch.Tensor,  # [B, S, D_img] -> bmm için
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
