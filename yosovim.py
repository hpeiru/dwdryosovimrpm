import torch
import torch.nn as nn


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class LayerNorm2D(nn.Module):
    def __init__(self, num_channels, eps=1e-5, affine=True):
        super(LayerNorm2D, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        var = x.var(dim=1, keepdim=True)  # (B, 1, H, W)

        x_normalized = (x - mean) / torch.sqrt(var + self.eps)  # (B, C, H, W)

        if self.affine:
            x_normalized = x_normalized * self.weight + self.bias

        return x_normalized


class MlpHead(nn.Module):
    def __init__(self, dim, num_classes=1000, mlp_ratio=4, act_type="relu", drop_rate=0.2):
        super().__init__()
        hidden_features = min(2048, int(mlp_ratio * dim))
        self.fc1 = nn.Linear(dim, hidden_features, bias=False)
        self.norm = nn.BatchNorm1d(hidden_features)
        self.act = Act(hidden_features, act_type)
        self.drop = nn.Dropout(drop_rate)
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

class Act(nn.Module):
    def __init__(self, out_planes=None, act_type="relu", inplace=True):
        super(Act, self).__init__()

        self.act = None
        if act_type == "relu":
            self.act = nn.ReLU(inplace=inplace)
        elif act_type == "prelu":
            self.act = nn.PReLU(out_planes)
        elif act_type == "hardswish":
            self.act = nn.Hardswish(inplace=True)
        elif act_type == "silu":
            self.act = nn.SiLU(inplace=True)
        elif act_type == "gelu":
            self.act = nn.GELU()

    def forward(self, x):
        if self.act is not None:
            x = self.act(x)
        return x


class SE(nn.Module):
    def __init__(self, dim, ratio=8):
        super().__init__()
        hidden_dim = max(8, dim // ratio)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, groups=1, kernel_size=3, stride=1, padding=None, act_type="gelu", use_bn=True):
        super(ConvX, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, groups=groups, padding=kernel_size//2 if padding is None else padding, bias=False)
        self.norm = nn.BatchNorm2d(out_planes) if use_bn else nn.Identity()
        self.act = Act(out_planes, act_type)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)
        return out


class CNNBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, act_type="gelu", kernel_size=3, mlp_ratio=1.0, use_se=True, drop_path=0.0):
        super(CNNBlock, self).__init__()
        mid_planes = int(out_planes*mlp_ratio)

        self.proj_in = ConvX(in_planes, mid_planes, groups=1, kernel_size=1, stride=1, act_type=act_type)
        self.gw = ConvX(mid_planes, mid_planes, groups=mid_planes//4, kernel_size=kernel_size, stride=stride, act_type=act_type)
        self.se = SE(mid_planes) if use_se else nn.Identity()
        self.proj_out = ConvX(mid_planes, out_planes, groups=1, kernel_size=1, stride=1, act_type=None)

        if stride == 2:
            self.skip = nn.Sequential(
                ConvX(in_planes, in_planes, groups=in_planes//4, kernel_size=3, stride=stride, act_type=None),
                ConvX(in_planes, out_planes, groups=1, kernel_size=1, stride=1, act_type=None)
            )
        else:
            self.skip = nn.Identity()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        out = self.proj_in(x)
        out = self.gw(out)
        out = self.se(out)
        out = self.proj_out(out)
        out = self.drop_path(out) + self.skip(x)
        return out


class HSMSSD(nn.Module):
    def __init__(self, planes, state_dim=64, drop_path=0.0):
        super().__init__()
        conv_dim = state_dim * 2

        self.ln = LayerNorm2D(planes)
        self.BCdt_proj = ConvX(planes, conv_dim, kernel_size=1, act_type=None, use_bn=False)
        self.gw = ConvX(conv_dim, conv_dim, groups=conv_dim//4, kernel_size=3, act_type=None, use_bn=False)
        self.out_proj = ConvX(planes, planes, kernel_size=1, act_type=None, use_bn=False)

        self.ssd = SSD(planes, state_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        BCdt = self.gw(self.BCdt_proj(self.ln(x)))
        y = self.ssd(BCdt, x)
        y = self.out_proj(y)
        x = self.drop_path(y) + x
        return x


class SSD(nn.Module):
    def __init__(self, planes, state_dim, A_init_range=(1, 16)):
        super().__init__()
        A = torch.empty(state_dim, dtype=torch.float32).uniform_(*A_init_range)
        self.A = torch.nn.Parameter(A)

    def forward(self, BCdt, x):
        BC, dt = torch.chunk(BCdt.flatten(2), chunks=2, dim=1)
        A = (dt + self.A.view(1, -1, 1)).softmax(-1)
        AB = (A * BC)
        h = x.flatten(2) @ AB.transpose(-2,-1)
        y = h @ BC
        y = y.reshape(*x.shape)
        return y


class YOSOViM(nn.Module):
    # pylint: disable=unused-variable
    def __init__(self, dims, layers, state_dims, act_type="gelu", mlp_ratio=1.0, kernel_size=3, drop_path_rate=0., num_classes=1000):
        super(YOSOViM, self).__init__()
        self.act_type = act_type
        self.mlp_ratio = mlp_ratio
        self.kernel_size = kernel_size
        self.drop_path_rate = drop_path_rate

        if isinstance(dims, int):
            dims = [dims//4, dims//2, dims, dims*2, dims*4]
        else:
            dims = [dims[0]//4, dims[0]//2] + dims

        self.first_conv = nn.Sequential(
            ConvX(3, dims[0], 1, 3, 2, act_type=act_type),
            ConvX(dims[0], dims[1], 1, 3, 2, act_type=act_type),
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(layers))]

        self.layer1 = self._make_layers(dims[1], dims[2], layers[0], 2, state_dims[0], drop_path=dpr[:layers[0]])
        self.layer2 = self._make_layers(dims[2], dims[3], layers[1], 2, state_dims[1], drop_path=dpr[layers[0]:sum(layers[:2])])
        self.layer3 = self._make_layers(dims[3], dims[4], layers[2], 2, state_dims[2], drop_path=dpr[sum(layers[:2]):sum(layers[:3])])

        head_dim = max(1024, dims[4])
        self.head = ConvX(dims[4], head_dim, 1, 1, 1, act_type=act_type)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = MlpHead(head_dim, num_classes, act_type=act_type)

        self.init_params(self)

    def _make_layers(self, inputs, outputs, num_block, stride, state_dim, drop_path):
        layers = [CNNBlock(inputs, outputs, stride, self.act_type, self.kernel_size, self.mlp_ratio, use_se=False, drop_path=drop_path[0])]

        for i in range(1, num_block):
            layers.append(CNNBlock(outputs, outputs, 1, self.act_type, self.kernel_size, self.mlp_ratio, use_se=True, drop_path=drop_path[i]))
            if state_dim > 0:
                layers.append(HSMSSD(outputs, state_dim, drop_path=drop_path[i]))

        return nn.Sequential(*layers)

    def init_params(self, model):
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        out = self.head(x)
        out = self.gap(out).flatten(1)
        out = self.classifier(out)
        return out

