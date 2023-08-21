import numpy as np
import torch
import torch.nn as nn
from utils.utils import EqualLinear

class Transformer(nn.Module):
    def __init__(
        self,
        num_layers,
        nhead,
        d_model,
        dim_feedforward,
        cls_type="cls_learn",
        pos_type="pos_learn",
        agg_method="mean",
        transformer_metric="dot_prod",
    ):
        super().__init__()

        self.cls_type = cls_type
        self.pos_type = pos_type
        self.agg_method = agg_method

        #self.mu_dec = nn.Linear(dim_feedforward, dim_feedforward)
        #self.adj_dec = nn.Linear(dim_feedforward, dim_feedforward)

        self.mu_dec = EqualLinear(dim_feedforward, dim_feedforward)
        self.adj_dec = EqualLinear(dim_feedforward, dim_feedforward)

        self.relu = nn.ReLU()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout=0.1,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # encoder_layer_stv = nn.TransformerEncoderLayer(
        #     d_model,
        #     nhead,
        #     dim_feedforward,
        #     dropout=0.1,
        #     activation="relu",
        # )
        # self.encoder_stv = nn.TransformerEncoder(encoder_layer_stv, num_layers)


    def forward(self, x, cls_tokens, adj_tokens):

        # 
        # # Concatenate cls tokens with support embeddings
        # if self.cls_type in ["cls_learn", "rand_const"]:
        #     cls_tokens = self.cls_embeddings(n_arng)  # (ways, dim)
        # elif self.cls_type == "proto":
        #     cls_tokens = gen_prototypes(x, ways, shot, self.agg_method)  # (ways, dim)
        # else:
        #     raise NotImplementedError

        # embeds = torch.cat((cls_tokens.unsqueeze(0), adj_tokens.unsqueeze(0), x), dim=0)

        # mu, stv = self.encoder(embeds)[:2]
        # mu = self.mu_dec(self.relu(mu))
        # stv = self.adj_dec(self.relu(stv))


        cls_sup_embeds = torch.cat((cls_tokens.unsqueeze(0), adj_tokens.unsqueeze(0), x), dim=0)  # (49+2,b, dim)
        # cls_sup_embeds = torch.unsqueeze(
        #     cls_sup_embeds, dim=1
        # )  # (ways*(shot+1), BS, dim)

        # Position embeddings based on class ID
        # pos_idx = torch.cat((n_arng, torch.repeat_interleave(n_arng, shot)))
        # pos_tokens = torch.unsqueeze(
        #     self.pos_embeddings(pos_idx), dim=1
        # )  # (ways*(shot+1), BS, dim)

        # Inputs combined with position encoding
        #transformer_input = cls_sup_embeds + pos_tokens
        

        mu, svd = self.relu(self.encoder(cls_sup_embeds)[:2]) # b x dim
        mu = self.mu_dec(mu)
        svd = self.adj_dec(svd) 


        # mu = self.relu(self.encoder(cls_sup_embeds)[0]) # b x dim
        # mu = self.mu_dec(mu) 

        

        # adj = x - mu.unsqueeze(0)
        # #adj_tokens.view(1,1,-1)
        # adj_sup_embeds = torch.cat((adj_tokens.unsqueeze(0), adj), dim=0) # (49+1,b, dim)

        # stv = self.relu(self.encoder_stv(adj_sup_embeds)[0])
        # #stv = self.relu(self.adj_dec(stv)) # b x dim
        # stv = self.relu(self.adj_dec(stv))


        return mu, svd