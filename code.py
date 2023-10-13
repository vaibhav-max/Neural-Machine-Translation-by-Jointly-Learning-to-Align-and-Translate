import torch
import torch.nn as nn
import numpy as np
import math

class InputEmbedding(nn.Module) :
    def __init__(self, d_model, vocab_size) :
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x) :
        x = self.embedding(x)
        x = x * math.sqrt(self.d_model)
        return x
    
class Encoder(nn.Module):
    def __init__(self, d_model, src_seq_size, encoder_hidden_size, num_layers = 1):
        super().__init__()
        self.num_layers = num_layers
        self.encoder_hidden_size = encoder_hidden_size
        self.seq_len = src_seq_size
        self.d_model = d_model
        self.encoder = nn.GRU(d_model, encoder_hidden_size, num_layers=self.num_layers, batch_first=True)
    
    def forward(self, x):
        encoder_ops, encoder_hidden = self.encoder(x)
        return encoder_ops, encoder_hidden[0]
    
class AttentionBlock(nn.Module):
    def __init__(self, hidden_size, d_ff) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.W_a = nn.Linear(in_features= hidden_size,out_features= d_ff)
        self.U_a = nn.Linear(in_features= 2*hidden_size,out_features= d_ff)
        self.V_a = nn.Linear(in_features= d_ff, out_features= 1)
        
    def forward(self, s, h):
        # s --> decoder hidden state
        # h --> encoder all hidden
        # print(s.shape, h.shape)
        energies = []
        for i in range(h.shape[1]):
            x = torch.tanh(self.W_a(s) + self.U_a(h[:,i,:]))
            x = self.V_a(x)
            energies.append(x)
        energy = torch.cat(energies, dim = 1)
        energy = energy.unsqueeze(2)
        weighted_h = energy * h
        context = torch.sum(weighted_h, dim=1)
        return context
    
class DecoderBlock(nn.Module):
    def __init__(self, hidden_size, d_model) :
        super().__init__()
        self.hidden_size = hidden_size
        self.d_model = d_model
        
        self.W = nn.Linear(in_features= d_model, out_features= hidden_size)
        self.W_z = nn.Linear(in_features= d_model, out_features= hidden_size)
        self.W_r = nn.Linear(in_features= d_model, out_features= hidden_size)
        
        self.C = nn.Linear(in_features= 2*hidden_size, out_features= hidden_size)
        self.C_z = nn.Linear(in_features= 2*hidden_size, out_features= hidden_size)
        self.C_r = nn.Linear(in_features= 2*hidden_size, out_features= hidden_size)
        
        self.U = nn.Linear(in_features= hidden_size, out_features= hidden_size)
        self.U_z = nn.Linear(in_features= hidden_size, out_features= hidden_size)
        self.U_r = nn.Linear(in_features= hidden_size, out_features= hidden_size)
        
    def forward(self, tgt, s, c) :
        z = torch.sigmoid(self.W_z(tgt) + self.U_z(s))
        r = torch.sigmoid(self.W_r(tgt) + self.U_r(s))
        r = r * s
        s_i = torch.tanh(self.W(tgt) + self.U(r) + self.C(c))
        s_i = (1-z)*s + z*s_i
        return s_i

class ProjectLayer(nn.Module):
    def __init__(self, tgt_vocab_size, hidden_size, d_model, l):
        super().__init__()
        self.hidden_size = hidden_size
        self.tgt_vocab_size = tgt_vocab_size
        self.l = l
        self.d_model = d_model
        
        self.U_0 = nn.Linear(in_features= hidden_size, out_features= 2*l)
        self.V_0 = nn.Linear(in_features= d_model, out_features= 2*l)
        self.C_0 = nn.Linear(in_features= 2*hidden_size, out_features= 2*l)
        
        self.maxPool1 = nn.MaxPool1d(kernel_size= 2, stride = 2)
        
        self.W_0 = nn.Linear(in_features= l, out_features= d_model)
        
    def forward(self, tgt, s, c, tgt_vocab) :
        t = self.U_0(s) + self.V_0(tgt) + self.C_0(c)
        t = self.maxPool1(t)
        t = self.W_0(t)
        t = torch.matmul(t, tgt_vocab.squeeze(0).T)
        return t
    
class Decoder(nn.Module) :
    def __init__(self,tgt_seq_len, hidden_size, decoderBlock, attention, project):
        super().__init__()
        self.linear = nn.Linear(in_features= 2*hidden_size, out_features= hidden_size)
        self.decoderBlock = decoderBlock
        self.attention = attention
        self.project = project
        self.tgt_seq_len = tgt_seq_len
        
    def forward(self, tgt_seq, tgt_seq_len ,s ,h, tgt_vocab) :
        # I need to return batch_size, seq_length, d_model
        batch_size, seq_length = tgt_seq.size(0), tgt_seq.size(1)
        vocab_size = tgt_vocab.size(1)
        x = torch.zeros(batch_size, seq_length, vocab_size)
        s = self.linear(s)
        for i in range(0, tgt_seq.size(1)):
            c = self.attention(s, h)
            s_i = self.decoderBlock(tgt_seq[:,i], s, c)
            y_i = self.project(tgt_seq[:, i], s, c, tgt_vocab)
            # this gives batch_size, d_model
            s = s_i
            x[:,i, :] = y_i
        return x
    
class Bahadanu_attention(nn.Module):
    def __init__(self,tgt_vocab_size, srcEmbedding : InputEmbedding, tgtEmbedding : InputEmbedding, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.srcEmbedding = srcEmbedding
        self.tgtEmbedding = tgtEmbedding
        self.encoder = encoder
        self.decoder = decoder
        self.tgt_vocab_size = tgt_vocab_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def encode(self, x):
        x = self.srcEmbedding(x)
        return self.encoder(x)
    
    def decode(self, x, tgt_seq_len, s, h):
        x = self.tgtEmbedding(x)
        tgt_indices = torch.arange(0, self.tgt_vocab_size).unsqueeze(0).to(self.device)
        tgt_vocab = self.tgtEmbedding(tgt_indices)
        return self.decoder(x, tgt_seq_len, s, h, tgt_vocab)
    
def getModel(src_seq_length, tgt_seq_length, src_vocab_size, tgt_vocab_size, d_model, encoder_hidden_size, decoder_hidden_size, d_ff, l):
    srcEmbedding = InputEmbedding(d_model= d_model, vocab_size= src_vocab_size)
    tgtEmbedding = InputEmbedding(d_model= d_model, vocab_size= tgt_vocab_size)
    
    encoder = Encoder(d_model= d_model, src_seq_size= src_seq_length, encoder_hidden_size= encoder_hidden_size, num_layers= 1)
    attention = AttentionBlock(hidden_size= decoder_hidden_size, d_ff= d_ff)
    decoderBlock = DecoderBlock(hidden_size= decoder_hidden_size, d_model= d_model)
    project = ProjectLayer(tgt_vocab_size= tgt_vocab_size, hidden_size= decoder_hidden_size, d_model= d_model, l= l)
    decoder = Decoder(tgt_seq_len= tgt_seq_length,hidden_size= decoder_hidden_size, decoderBlock= decoderBlock, attention= attention, project= project)
    
    return Bahadanu_attention(tgt_vocab_size = tgt_vocab_size,
                              srcEmbedding= srcEmbedding,
                              tgtEmbedding= tgtEmbedding,
                              encoder= encoder,
                              decoder= decoder
                              )
            

        