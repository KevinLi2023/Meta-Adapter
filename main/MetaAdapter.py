import torch
from torch import nn
import timm
import math


def forward_block_MetaAdapter(self, x):
    x = x + self.drop_path(self.attn(self.norm1(x)))
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    if self.metaAdapter is not None:
      x = x + self.drop_path(self.metaAdapter(self.norm0(x))) * self.s
    return x

def forward_swin_block_MetaAdapter(self, x):
    H, W = self.input_resolution
    B, L, C = x.shape
    assert L == H * W, "input feature has wrong size"

    shortcut = x + self.drop_path(self.adapter_attn(self.norm1(x))) * self.s
    x = self.norm1(x)
    x = x.view(B, H, W, C)

    # cyclic shift
    if self.shift_size > 0:
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
    else:
        shifted_x = x

    # partition windows
    x_windows = timm.models.swin_transformer.window_partition(shifted_x,
                                                              self.window_size)  # nW*B, window_size, window_size, C
    x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

    # W-MSA/SW-MSA
    attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

    # merge windows
    attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
    shifted_x = timm.models.swin_transformer.window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

    # reverse cyclic shift
    if self.shift_size > 0:
        x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
    else:
        x = shifted_x
    x = x.view(B, H * W, C)

    # FFN
    x = shortcut + self.drop_path(x)
    x = x + self.drop_path(self.mlp(self.norm2(x)))
  
    if self.metaAdapter is not None:
      x = x + self.drop_path(self.metaAdapter(self.norm0(x))) * self.s

    return x


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ConvAdapter(nn.Module):
    def __init__(self, dim=8, xavier_init=False):
        super().__init__()
        self.adapter_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=4)
        if xavier_init:
            nn.init.xavier_uniform_(self.adapter_conv.weight)
        else:
            nn.init.zeros_(self.adapter_conv.weight)
            self.adapter_conv.weight.data[:, :, 1, 1] += torch.eye(1, dtype=torch.float)
        nn.init.zeros_(self.adapter_conv.bias)

        self.adapter_down = nn.Linear(768, dim)  # equivalent to 1 * 1 Conv
        self.adapter_up = nn.Linear(dim, 768)  # equivalent to 1 * 1 Conv
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        B, N, C = x.shape

        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        x_down = self.act(x_down)

        x_patch = x_down[:, 1:].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        x_patch = self.adapter_conv(x_patch)
        x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)

        x_cls = x_down[:, :1].reshape(B, 1, 1, self.dim).permute(0, 3, 1, 2)
        x_cls = self.adapter_conv(x_cls)
        x_cls = x_cls.permute(0, 2, 3, 1).reshape(B, 1, self.dim)

        x_down = torch.cat([x_cls, x_patch], dim=1)

        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv

        return x_up

class AttnAdapter(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1, Prompt_Token_num=10):
        super(AttnAdapter, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** (-0.5)
        self.drop=nn.Dropout(dropout)
        self.Prompt_Tokens_k = nn.Parameter(torch.zeros(1,int(Prompt_Token_num/2),dim))
        self.Prompt_Tokens_v = nn.Parameter(torch.zeros(1,int(Prompt_Token_num/2),dim))
        #initialization
        nn.init.xavier_uniform_(self.Prompt_Tokens_k)
        nn.init.xavier_uniform_(self.Prompt_Tokens_v)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim] 
        B0, N0, C0 = x.shape
        B1, N1, C1=self.Prompt_Tokens_k.shape

        q = x.reshape(B0, N0, self.num_heads, C0 // self.num_heads).permute(0, 2, 1, 3)
        k=self.Prompt_Tokens_k.reshape(B1, N1, self.num_heads, C1 // self.num_heads).permute(0, 2, 1, 3)
        v =self.Prompt_Tokens_v.reshape(B1, N1, self.num_heads, C1 // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.drop(attn.softmax(dim=-1))
        output=(attn @ v).transpose(1, 2).reshape(B0, N0, C0)
        return output

class MLPAdapter(nn.Module):
    def __init__(self,
                 dim,
                 dropout=0.0,
                 adapter_scalar=1.0,
                 down_size=8,
                 ):
        super().__init__()
        self.n_embd = dim
        self.down_size = down_size
        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = adapter_scalar
        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)
        self.dropout = dropout
        nn.init.kaiming_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout)
        up = self.up_proj(down)
        return up

class MetaAdapter_swin(nn.Module):
    def __init__(self, dim=8, xavier_init=True, vit_dim=768):
        super().__init__()
        self.adapter_conv = nn.Conv2d(dim, dim, 3, 1, 1)
        if xavier_init:
            nn.init.xavier_uniform_(self.adapter_conv.weight)
        else:
            nn.init.zeros_(self.adapter_conv.weight)
            self.adapter_conv.weight.data[:, :, 1, 1] += torch.eye(8, dtype=torch.float)
        nn.init.zeros_(self.adapter_conv.bias)

        self.adapter_down = nn.Linear(vit_dim, dim)
        self.adapter_up = nn.Linear(dim, vit_dim)
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)
        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        B, N, C = x.shape
        H = int(math.sqrt(N))
        x_down = self.adapter_down(x)
        x_patch = x_down.reshape(B, H, H, self.dim).permute(0, 3, 1, 2)
        x_patch = self.act(x_patch)
        x_patch = self.adapter_conv(x_patch)
        x_down = x_patch.permute(0, 2, 3, 1).reshape(B, -1, self.dim)
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)

        return x_up


def set_MetaAdapter(model, method, adapter, dim=8, s=1, xavier_init=False):
    if method == 'MetaAdapter':
        for _ in model.children():
            if type(_) == timm.models.vision_transformer.Block:
                if adapter == 'ConvAdapter':
                  _.norm0 = nn.LayerNorm(768, eps=1e-6)
                  _.metaAdapter = ConvAdapter(dim)
                  _.s = s
                  bound_method =  forward_block_MetaAdapter.__get__(_, _.__class__)
                  setattr(_, 'forward', bound_method)
                elif adapter == 'AttnAdapter':
                  _.norm0 = nn.LayerNorm(768, eps=1e-6)
                  _.metaAdapter = AttnAdapter(dim=768)
                  _.s = s
                  bound_method =  forward_block_MetaAdapter.__get__(_, _.__class__)
                  setattr(_, 'forward', bound_method)
                elif adapter == 'MLPAdapter':
                  _.norm0 = nn.LayerNorm(768, eps=1e-6)
                  _.metaAdapter = MLPAdapter(dim=768, down_size=dim)
                  _.s = s
                  bound_method =  forward_block_MetaAdapter.__get__(_, _.__class__)
                  setattr(_, 'forward', bound_method)
                
            elif len(list(_.children())) != 0:
              set_MetaAdapter(_, method, adapter, dim, s, xavier_init)
