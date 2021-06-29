import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """ split image into patches and then embed them.

    Parameters
    __________
    img_size : int
        size of the image (it is a square).

    in_chans : int
        Number of input channels.

    embed_dim : int
        The embedding dimension.

    Attributes
    __________
    n_patches : int
        Number of patches inside of our image.

    proj : nn.Conv2d
        Convolutional layer that does both the splitting into patches and their embedding
    """

    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    def forward(self, x):
        """Run forward pass.
        Parameters
        __________
        x : torch.Tensor
            Shape `(n_shape, in_chans, img_size, img_size)`.
        Returns
        _______
        torch.Tensor
            Shape `(n_samples, n_patches, embed_dim)`.
        """
        x=self.proj(
            x
        ) # (n_samples, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(start_dim=2) # (n_samples, embed_dim, n_patches)
        x = x.transpose(1,2) # (n_samples. n_patches, embed_dim)
        return x

# This class is basically the encoder from the Attention is All You Need Paper.
class Attention(nn.Module):
    """Attention mechanism.

    Parameters
    __________
    dim : int
        The input and out dimension of per token features.

    n_head : int
        Number of attention heads.

    qkv_bias : bool
        If True then we include bias to the query, key and value projections

    attn_p : float
        Dropout probability applied to the query, key and value tensors.

    proj_p : float
        Dropout probability applied to the output tensor.


    Attributes
    __________
    scale : float
        Normalizing constant for the dot product.

    qkv : nn.Linear
        Linear projection for the query, key and value.

    proj : nn.Linear
        Linear mapping that takes int the concatenated output of all attention heads and maps it into a new space.

    attn_drop, proj_drop : nn. Dropout
        Dropout layers.
    """
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads # so that when we concat all heads, will have same dimensionality of dim
        self.scale = self.head_dim ** -0.5 # do not feed huge values into softmax, can lead to small gradients
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        """Run forward pass.

        Parameters
        __________
        x : torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`. Note the + 1 is due to the CLS embedding

        Returns
        _______
        torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.
        """
        n_samples, n_tokens, dim = x.shape

        # sanity check
        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x) # (n_samples, n_patches + 1, 3*dim)
        qkv = qkv.reshape(
            n_samples, n_tokens, 3, self.n_heads, self.head_dim
        ) # (n_samples, n_patches + 1, 3, n_heads, head_dim) 3 for key query, value. Rest is for multi head.
        qkv = qkv.permute(
            2,0,3,1,4
        )# change order (3, n_samples, n_heads, n_patches + 1, head_dim)
        # there are n_patches + 1 patches of dimension head_dim for query and key. These are matmul to get attention.
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Transpose keys to get dot product
        k_t = k.transpose(-2,-1) # (n_samples, n_heads, head_dim, n_patches + 1)

        dp = (
            q @ k_t # last 2 dimensions are compatible
        ) * self.scale # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = dp.softmax(dim = -1) # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = self.attn_drop(attn)
        weighted_avg = attn @ v # (n_samples, n_heads, n_patches + 1, head_dim)
        weighted_avg = weighted_avg.transpose(1, 2) # (n_samples, n_patches + 1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2) # (n_samples, n_patches + 1, dim) # concat attention heads
        x = self.proj(weighted_avg)
        x = self.proj_drop(x)

        return x

class MLP(nn.Module):
    """Multilayer Perceptron

    Parameters
    __________
    in_features : int
        Number of input features.

    hidden_features : int
        Number of nodes in the hidden layer.

    out_features : int
        Number of output features.

    p : float
        Dropout probability.

    Attributes
    __________
    fc : nn.Linear
        The First linear layer.

    act : nn.GELU
        GELU activation function. Gaussian Error Linear Unit Activation Function

    fc2 : nn.Linear
        The second linear layer.

    """
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        """Run forward pass.

        Parameters
        __________
        x : torch.Tensor
            shape `(n_samples, n_patches + 1, in_features)'.

        Returns
        _______
        torch.Tensor
            Shape `(n_samples, n_patches + 1, out_features)`
        """
        x = self.fc1(
            x
        ) # (n_samples, n_patches + 1, hidden_features)
        x = self.act(x) # (n_samples, n_patches + 1, hidden_features)
        x = self.drop(x)
        x = self.fc2(x) # (n_samples, n_patches + 1, hidden_features)
        x = self.drop(x) # (n_samples, n_patches + 1, hidden_features)

        return x

class Block(nn.Module):
    """Transofrmer block.

    Parameters
    __________
    dim : int
        Embedding dimension.

    n_heads : int
        Number of attention heads.

    mlp_ratio : float
        Determines the hidden dimension size of the `MLP` module with respect to `dim`.

    qkv_bias : bool
        If True then we include bias to the query, key, and value projections.

    p, attn_p : float
        Dropout probability.

    Attributes
    __________
    norm1, norm2 : LayerNorm
        Layer normalization.

    attn :  Atention
        Attention module.

    mlp : MLP
        MLP module.
    """
    def __init__(self, dim, n_heads, mlp_ratio=3.0, qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            proj_p=p
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features = hidden_features,
            out_features=dim
        )

    def forward(self, x):
        """Run forward pass.
        Parameters
        __________
        x : torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.
        Returns
        _______
        torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.
        """
        x = x + self.attn(self.norm1(x)) # different layer norm, each has diff parms
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    """Simplified implementation of Vision Transformer
    Parameters
    __________
    img_size : int
        Both height and width of image (square)
    patch_size : int
        Both height and width of patch (square)
    in_chans : int
        Number of input channels.
    n_classes : int
        Number of classes.
    embed_dim : int
        Dimensionality of the token/patch embeddings.
    depth : int
        Number of blocks.
    n_heads : int
        Number of attention heads.
    mlp_ratio : float
        Hidden dimensions of the MLP module
    qkv_bias : bool
        If true then include bias into the query key and value projections
    p, attn_p : float
        Dropout probability.
    Attributes
    __________
    patch_embed : PatchEmbed
        Instance of PatchEmbed Layer
    class_token : nn.Parameter
        Learnable parameter that will represent the first token in the sequence.
        It has 'embed_dim' elements
    pos_emb : nn.Parameter
        Positional embedding of the cls token + all the patches.
        It has '(n_patches + 1) * embed_dim' elements.
    pos_drop : nn.Dropout
        Dropout Layer
    blocks : nn.ModuleList
        List of block modules (hold all block modules)
    norm : nn.LayerNorm
        Layer normalization
    """
    def __init__(self,
                img_size = 384,
                patch_size = 16,
                in_chans = 3,
                n_classes=1000,
                embed_dim=768,
                depth=12,
                n_heads=12,
                mlp_ratio=4.,
                qkv_bias=True,
                p=0.,
                attn_p=0.,
                ):
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim)) # first two dims for convenience
        # +1 in n patches because of the class token
        self.pos_embed = nn.Parameter(torch.zeros(1,1+self.patch_embed.n_patches, embed_dim)) # learnable locations
        self.pos_drop = nn.Dropout(p=p)
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                p=p,
                attn_p=attn_p
            )
            for _ in range(depth) # iteratively create transformer encoder.
            # All hyperparams in block are the same, but each block will have its own parameters
        ])

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        """Run forward pass
            Parameters
            __________
            x : torch.Tensor
                Shape `(n_samples, in_chans, img_size, img_size)
            Returns
            _______
            logits : torch.Tensor
                Logits over all the classes - `(n_samples, n_classes)`.
        """
        n_samples = x.shape[0]
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand( # take class token, replicate over sample dimension
            n_samples, -1, -1
        ) # (n_samples, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1) # (n_samples, 1 + n_patches, embed_dim)
        x = x + self.pos_embed # (n_samples, 1+n_patches, embed_dim) (broadcasting done)
        x = self.pos_drop(x)

        for block in self.blocks: # iteratively define blocks in encoder
            x = block(x)

        x = self.norm(x)

        cls_token_final = x[:, 0] # just the CLS token
        x = self.head(cls_token_final)
        return x


