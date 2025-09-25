import torch
import torch.nn as nn


class SoftMaxLayers(nn.Module):
    def __init__(self, d_model=256, ffn_dim_expansion=4, head_num=1, ffn=True):
        super(SoftMaxLayers, self).__init__()

        self.USE_ffn = ffn

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.Linear = nn.Linear(d_model, d_model, bias=False)
        self.norm1 = nn.LayerNorm(d_model)

        if self.USE_ffn:
            self.ffn = nn.Sequential(nn.Linear(d_model * 2, d_model * 2 * ffn_dim_expansion),
                                     nn.GELU(),
                                     nn.Linear(d_model * 2 * ffn_dim_expansion, d_model))
            self.norm2 = nn.LayerNorm(d_model)

    def forward(self, source, target, h, w):
        # print("Soft")

        query, key, value = source, target, target

        query = self.q_proj(query)  # [B, L, C]
        key = self.k_proj(key)  # [B, L, C]
        value = self.v_proj(value)  # [B, L, C]

        b, _, c = query.size()
        query = query.view(b, h, w, c)
        key = key.view(b, h, w, c)
        value = value.view(b, h, w, c)

        scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / (c ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, value).view(b, -1, c)  # [B, L, C]

        message = self.Linear(out)
        message = self.norm1(message)
        if self.USE_ffn:
            message = self.ffn(torch.cat([source, message], dim=-1))
            message = self.norm2(message)

        return source + message


class StereoLayers(nn.Module):
    def __init__(self, d_model=256, ffn_dim_expansion=4, head_num=1, ffn=True):
        super(StereoLayers, self).__init__()

        self.USE_ffn = ffn
        self.head_num = head_num

        self.d_model = d_model

        self.dp_head = d_model // head_num
        assert (
                self.dp_head * self.head_num == self.d_model
        ), "Embedding dimension must be divisible by the number of heads"

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.Linear = nn.Linear(d_model, d_model, bias=False)
        self.norm1 = nn.LayerNorm(d_model)

        if self.USE_ffn:
            self.ffn = nn.Sequential(nn.Linear(d_model * 2, d_model * 2 * ffn_dim_expansion),
                                     nn.GELU(),
                                     nn.Linear(d_model * 2 * ffn_dim_expansion, d_model))
            self.norm2 = nn.LayerNorm(d_model)

    def forward(self, source, target, h, w):

        B, L, C = source.shape

        query, key, value = source, target, target

        query = self.q_proj(query)  # [B, L, C]
        key = self.k_proj(key)  # [B, L, C]
        value = self.v_proj(value)  # [B, L, C]

        assert query.shape[1] == L == h * w
        assert self.d_model == C

        query = query.view(B, L, self.head_num, self.dp_head)  # [B, L, N, D]
        key = key.view(B, L, self.head_num, self.dp_head)
        value = value.view(B, L, self.head_num, self.dp_head)


        # query = query.reshape(B, h, w, self.head_num, self.dp_head).reshape(B * h, w, self.head_num, self.dp_head)
        # key = key.reshape(B, h, w, self.head_num, self.dp_head).reshape(B * h, w, self.head_num, self.dp_head)
        # value = value.reshape(B, h, w, self.head_num, self.dp_head).reshape(B * h, w, self.head_num, self.dp_head)

        key = key.softmax(dim=1)

        context = torch.einsum("bnhd, bnhe->bhde", query, key) # / (L ** 0.5)  # [B, N, D, D] Q K
        # context = context.softmax(dim=-1)
        # print(context.shape)
        attn = torch.einsum("bhde, bnhe->bnhd", context, value)
        # print(attn.shape)

        attn = attn.contiguous().view(B, L, C)
        # print(attn.shape)

        message = self.Linear(attn)
        message = self.norm1(message)

        if self.USE_ffn:
            message = self.ffn(torch.cat([source, message], dim=-1))
            message = self.norm2(message)

        return source + message


class TransformerBlock(nn.Module):
    def __init__(self, d_model=256, head_num=1, mode='Stereo', ffn_dim_expansion=4):
        super(TransformerBlock, self).__init__()

        self.d_model = d_model

        self.mode = mode

        if self.mode == 'SoftMax':

            # self.self_attn = SoftMaxLayers(d_model=self.d_model, head_num=head_num, ffn_dim_expansion=ffn_dim_expansion,
            #                             ffn=False)
            self.cross_attn = SoftMaxLayers(d_model=self.d_model, head_num=head_num,
                                            ffn_dim_expansion=ffn_dim_expansion)

        elif self.mode == 'Stereo':

            # self.self_attn = StereoLayers(d_model=self.d_model, head_num=head_num, ffn_dim_expansion=ffn_dim_expansion,
            #                            ffn=False)
            self.cross_attn = StereoLayers(d_model=self.d_model, head_num=head_num, ffn_dim_expansion=ffn_dim_expansion)

    def forward(self, source, target, h, w):
        # source, target: [B, L, C]
        # self attention
        # source = self.self_attn(source, source, h, w)
        # cross attention and ffn
        source = self.cross_attn(source, target, h, w)
        return source


class FeatureTransformer(nn.Module):
    def __init__(self, num_layers=1, d_model=256, head_num=1, mode='Stereo', ffn_dim_expansion=4):
        super(FeatureTransformer, self).__init__()
        self.d_model = d_model

        self.layers = nn.ModuleList([
            TransformerBlock(d_model=d_model,
                             head_num=head_num,
                             mode=mode,
                             ffn_dim_expansion=ffn_dim_expansion)
            for i in range(num_layers)])

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feature_0, feature_1):

        b, c, h, w = feature_0.shape

        assert self.d_model == c

        feat0 = feature_0.flatten(-2).permute(0, 2, 1)  # [B, H*W, C]
        feat1 = feature_1.flatten(-2).permute(0, 2, 1)  # [B, H*W, C]
        concat0 = torch.cat((feat0, feat1), dim=0)  # [2B, H*W, C]
        concat1 = torch.cat((feat1, feat0), dim=0)  # [2B, H*W, C]

        for layer in self.layers:
            concat0 = layer(concat0, concat1, h, w)

            # update feature1
            concat1 = torch.cat(concat0.chunk(chunks=2, dim=0)[::-1], dim=0)

        feat0, feat1 = concat0.chunk(chunks=2, dim=0)  # [B, H*W, C]

        # reshape back
        feat0 = feat0.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        feat1 = feat1.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

        return feat0, feat1




if __name__ == '__main__':
    B = 1
    C = 2
    H = 3
    W = 4
    tensor0 = torch.rand(B, C, H, W)
    tensor1 = torch.rand(B, C, H, W)

    k = tensor0.reshape(B, C, H * W).permute(0, 2, 1)
    print(torch.softmax(k, dim=-1))
    print(torch.softmax(k, dim=1))

    transformer = FeatureTransformer(num_layers=1, d_model=C, head_num=1, mode='Stereo', ffn_dim_expansion=4)
    feat0, feat1 = transformer(tensor0, tensor1)

