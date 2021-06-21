from vit_pytorch import ViT
from TransGan_pytorch import models as TGModels

class autoencoder:
    # TODO: Set up configurations
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    def get_encoder(self, encoder):
        return encoder
    def get_decoder(self, decoder):
        return decoder

class tmpViT:
    def __init__(self, image_size, patch_size, num_classes, dim, depth,
                 heads, mlp_dim, dropout, emb_dropout, keep_head):

        self.ViT = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            emb_dropout=emb_dropout,
            keep_head=keep_head
        )

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.emb_dropout = emb_dropout
        self.keep_head = keep_head

    def get_ViT(self):
        return self.ViT

class tmpTransGan:
    def __init__(self, depth1, depth2, depth3, initial_size, dim, heads, mlp_ratio, drop_rate):
        self.depth1 = depth1
        self.depth2 = depth2
        self.depth3 = depth3
        self.initial_size = initial_size
        self.dim = dim
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.TransGan = TGModels.Generator(depth1=depth1, depth2=depth2, depth3=depth3, initial_size=initial_size,
                                           dim=dim, heads=heads, mlp_ratio=mlp_ratio, drop_rate=drop_rate)
    def get_TransGan(self):
        return self.TransGan

