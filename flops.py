import torch, math
from thop import profile, clever_format
from yosovim import YOSOViM, SSD


def count_ssd_cell(m: SSD, x: torch.Tensor, y: torch.Tensor):
    B, _S, H, W = x[0].shape
    _, C, _, _ = x[1].shape
    S = _S // 3

    m.total_ops += B * S * H * W
    m.total_ops += 3 * B * S * H * W
    m.total_ops += B * S * H * W
    m.total_ops += B * C * H * W * S
    m.total_ops += B * C * S * C
    m.total_ops += B * C * S
    m.total_ops += B * C * S * H * W

if __name__=="__main__":
    custom_ops = {
        SSD: count_ssd_cell
    }
    input = torch.randn(1, 3, 256, 256)

    #model = YOSOViM(dims=[64,128,256], layers=[2,2,2], state_dims=[32,16,8], mlp_ratio=3.0, kernel_size=3, act_type="gelu")
    #model = YOSOViM(dims=[64,128,256], layers=[3,4,3], state_dims=[32,16,8], mlp_ratio=3.0, kernel_size=3, act_type="gelu")
    #model = YOSOViM(dims=[80,160,320], layers=[3,7,3], state_dims=[32,16,8], mlp_ratio=3.0, kernel_size=3, act_type="gelu")
    model = YOSOViM(dims=[96,192,384], layers=[3,7,3], state_dims=[32,16,8], mlp_ratio=3.0, kernel_size=3, act_type="gelu")


    model.eval()
    print(model)
    
    macs, params = profile(model, inputs=(input, ), custom_ops=custom_ops)
    macs, params = clever_format([macs, params], "%.3f")
    
   # params = sum(p.numel() for p in model.parameters()) / 1e6
    print('Flops:  ', macs)
    print('Params: ', params)

