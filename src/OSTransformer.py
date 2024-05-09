import torch
import torch.nn.functional as F

__all__ = ["SurvivalWrapper"]

def dichotomize_n_mask_torch(sample: torch.Tensor):
    output_sample = torch.hstack((torch.eye(len(sample)).to(sample.device), torch.unsqueeze(sample, dim=1))).float()
    mask = torch.clone(sample)
    mask[~torch.isnan(sample)] = 1
    mask[torch.isnan(sample)] = 0

    output_sample[torch.isnan(output_sample)] = 0

    return output_sample, mask


class MlpBlock(torch.nn.Module):

    def __init__(self, in_dim, mlp_dim, out_dim, dropout_rate=0.1):
        super(MlpBlock, self).__init__()

        self.fc1 = torch.nn.Linear(in_dim, mlp_dim)
        self.fc2 = torch.nn.Linear(mlp_dim, out_dim)
        self.act = torch.nn.GELU()
        if dropout_rate > 0.0:
            self.dropout1 = torch.nn.Dropout(dropout_rate)
            self.dropout2 = torch.nn.Dropout(dropout_rate)
        else:
            self.dropout1 = None
            self.dropout2 = None

    def forward(self, x):

        out = self.fc1(x)
        out = self.act(out)
        if self.dropout1:
            out = self.dropout1(out)

        out = self.fc2(out)
        out = self.dropout2(out)

        return out


class LinearGeneral(torch.nn.Module):
    def __init__(self, in_dim=(768,), feat_dim=(12, 64)):
        super(LinearGeneral, self).__init__()

        self.weight = torch.nn.Parameter(torch.randn(*in_dim, *feat_dim))
        self.bias = torch.nn.Parameter(torch.zeros(*feat_dim))

    def forward(self, x, dims):
        a = torch.tensordot(x, self.weight, dims=dims) + self.bias
        return a


class SelfAttention(torch.nn.Module):
    def __init__(self, in_dim, heads=8, dropout_rate=0.1):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.head_dim = in_dim // heads
        self.scale = self.head_dim ** 0.5

        self.query = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.key = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.value = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.out = LinearGeneral((self.heads, self.head_dim), (in_dim,))

        if dropout_rate > 0:
            self.dropout = torch.nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x, mask):
        b, n, _ = x.shape

        q = self.query(x, dims=([2], [0]))
        k = self.key(x, dims=([2], [0]))
        v = self.value(x, dims=([2], [0]))

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        mask = torch.unsqueeze(mask, dim=2)
        mask = torch.repeat_interleave(mask, x.shape[1], dim=2)
        mask_transposed = torch.transpose(mask, 1, 2)
        mask = torch.mul(mask, mask_transposed)
        mask = torch.unsqueeze(mask, dim=1)
        mask = torch.repeat_interleave(mask, self.heads, dim=1)

        attn_weights = (torch.matmul(q, k.transpose(-2, -1))) / self.scale
        mask = mask == 0

        attn_weights = attn_weights.masked_fill(mask, -torch.inf)

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights)
        out = torch.matmul(attn_weights, v)
        out = out.permute(0, 2, 1, 3)

        out = self.out(out, dims=([2, 3], [0, 1]))

        return out


class EncoderBlock(torch.nn.Module):
    def __init__(self, in_dim, mlp_dim, num_heads, dropout_rate=0.1, attn_dropout_rate=0.1):
        super(EncoderBlock, self).__init__()

        self.norm1 = torch.nn.LayerNorm(in_dim)
        self.attn = SelfAttention(in_dim, heads=num_heads, dropout_rate=attn_dropout_rate)
        if dropout_rate > 0:
            self.dropout = torch.nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        self.norm2 = torch.nn.LayerNorm(in_dim)
        self.mlp = MlpBlock(in_dim, mlp_dim, in_dim, dropout_rate)

    def forward(self, x, mask):
        residual = x
        out = self.norm1(x)
        out = self.attn(out, mask)
        if self.dropout:
            out = self.dropout(out)
        out += residual
        residual = out

        out = self.norm2(out)
        out = self.mlp(out)
        out += residual
        return out


class Encoder(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dim, num_layers=12, num_heads=12, dropout_rate=0.1, attn_dropout_rate=0.0):
        super(Encoder, self).__init__()

        in_dim = emb_dim
        self.encoder_layers = torch.nn.ModuleList()
        for i in range(num_layers):
            layer = EncoderBlock(in_dim, mlp_dim, num_heads, dropout_rate, attn_dropout_rate)
            self.encoder_layers.append(layer)
        self.norm = torch.nn.LayerNorm(in_dim)

    def forward(self, x, mask):

        for layer in self.encoder_layers:
            x = layer(x, mask)

        out = self.norm(x)
        return out


class OSTransformer(torch.nn.Module):

    def __init__(self,
                 emb_dim=100,
                 mlp_dim=3072,
                 num_heads=12,
                 num_layers=12,
                 output_size=2,
                 attn_dropout_rate=0.0,
                 dropout_rate=0.1):
        super(OSTransformer, self).__init__()

        self.embedding = dichotomize_n_mask_torch

        self.transformer = Encoder(
            emb_dim=emb_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate)

        if dropout_rate > 0:
            self.dropout = torch.nn.Dropout(dropout_rate)
        else:
            self.dropout = None

        self.input_size = emb_dim - 1

        self.output_size = output_size
        self.classifier = torch.nn.Linear(emb_dim, self.output_size)

    def forward(self, x):
        emb = mask = torch.Tensor().to(x.device)

        for i, sample in enumerate(x):
            sample_emb, sample_mask = self.embedding(sample)
            emb = torch.cat([emb, sample_emb.unsqueeze(dim=0)], dim=0)
            mask = torch.cat([mask, sample_mask.unsqueeze(dim=0)], dim=0)

        b, *_ = emb.shape

        feat = self.transformer(emb, mask)

        logits = self.classifier(feat.mean(dim=1))

        return logits

########################################################################################################################

class CustomMLP(torch.nn.Module):

    def __init__(self, input_size: int, output_size: int, hidden_sizes: list = list, drop_rate: float = None):
        super(CustomMLP, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        if drop_rate is None:
            self.input = torch.nn.Sequential(torch.nn.Linear(input_size, hidden_sizes[0]), torch.nn.ReLU())
        else:
            self.input = torch.nn.Sequential(torch.nn.Linear(input_size, hidden_sizes[0]), torch.nn.ReLU(), torch.nn.Dropout(drop_rate))

        self.hidden = torch.nn.ModuleList()
        if len(hidden_sizes) > 1:
            for k in range(len(hidden_sizes) - 1):
                if drop_rate is not None and k < len(hidden_sizes) - 2:
                    layer = torch.nn.Sequential(torch.nn.Linear(hidden_sizes[k], hidden_sizes[k + 1]), torch.nn.ReLU(), torch.nn.Dropout(drop_rate))
                else:
                    layer = torch.nn.Sequential(torch.nn.Linear(hidden_sizes[k], hidden_sizes[k + 1]), torch.nn.ReLU())

                self.hidden.append(layer)

        self.output = torch.nn.Sequential(torch.nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, inputs):

        x = self.input(inputs)

        if self.hidden:
            for layer in self.hidden:
                x = layer(x)

        out = self.output(x)

        return out

########################################################################################################################

class SurvivalWrapper(torch.nn.Module):

    def __init__(self, num_events: int, max_time: int, shared_net_params: dict, cs_subnets_params: dict):
        super(SurvivalWrapper, self).__init__()

        self.shared_net = OSTransformer(**shared_net_params)

        self.input_size = self.shared_net.input_size
        self.output_size = ( num_events, max_time )
        self.max_time = max_time

        cs_subnets_params["input_size"] = self.shared_net.output_size
        cs_subnets_params["output_size"] = max_time

        self.CS_subnets = torch.nn.ModuleList()
        for k in range(num_events):
            subnet = CustomMLP(**cs_subnets_params)
            self.CS_subnets.append(subnet)

    def forward(self, inputs):

        x = self.shared_net(inputs)

        y = []
        for subnet in self.CS_subnets:
            x_CS = subnet(x)
            y_CS = F.softmax(x_CS, dim=-1)
            y.append(y_CS)

        y = torch.cat(y, dim=-1)

        return y


if __name__ == "__main__":
    pass
