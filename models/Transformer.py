from Attention import MultiHeadedAttention
from PositionwiseFeedForward import PositionwiseFeedForward
from PositionalEncoding import PositionalEncoding
from EncoderDecoder import EncoderDecoder
from Encoder import EncoderLayer, Encoder
from Decoder import DecoderLayer, Decoder
from Embeddings import Embeddings
from Generator import Generator
from AddNorm import LayerNorm, ResidualNet


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy

    attn = MultiHeadedAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    
    encoder_layer = EncoderLayer(d_model, c(attn), c(ff), dropout)
    decoder_layer = DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout)

    encoder = Encoder(encoder_layer, N)    
    decoder = Decoder(decoder_layer, N)
    
    src_embedding = nn.Sequential(Embeddings(d_model, src_vocab), c(position))
    tgt_embedding = nn.Sequential(Embeddings(d_model, tgt_vocab), c(position))
    
    generator = Generator(d_model, tgt_vocab)
    
    model =  EncoderDecoder(encoder, decoder, src_embedding, tgt_embedding, generator)
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    
    return model

