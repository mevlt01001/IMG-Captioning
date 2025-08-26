import torch

class LSTMDecoder(torch.nn.Module):
    
    def __init__(self, 
                 vocap_size:int, 
                 seq_len:int, 
                 input_feats:int, 
                 hidden_feats:int, 
                 num_layers:int,
                 device=torch.device('cpu')
                 ):
        super().__init__()
        self.vocap_size = vocap_size
        self.seq_len = seq_len
        self.input_feats = input_feats
        self.hidden_feats = hidden_feats
        self.num_layers = num_layers

        self.LSTM = torch.nn.LSTM(input_feats, hidden_feats, num_layers, batch_first=True, dropout=0.3).to(device=device)
        self.linearHead = torch.nn.Linear(hidden_feats, vocap_size).to(device=device)
    

    def forward(self, x):
        # x.shape is [B, seq, feats]:[1, 50, 512]
        out, (ht, ct) = self.LSTM.forward(x)
        out = self.linearHead(out)
        return out