import torch
import copy
import torch.nn as nn
from torchvision.transforms.functional import pil_to_tensor
from torchvision.datasets.folder import pil_loader
from model import SimpleCNN

IMAGE_SIZE = 31
HEAD_BLOCK = f"""#define IMAGE_H {IMAGE_SIZE}
#define IMAGE_W {IMAGE_SIZE}
#define KERNEL_H 3
#define KERNEL_W 3
"""
CONV_BLOCK = """#define CONV_{}_IN_CHANNELS {}
#define CONV_{}_OUT_CHANNELS {}
float conv_{}_weight[CONV_{}_OUT_CHANNELS][CONV_{}_IN_CHANNELS][KERNEL_H][KERNEL_W] = {{ {} }};
float conv_{}_bias[CONV_{}_OUT_CHANNELS] = {{ {} }};
"""


def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """将BatchNorm参数融合到Conv2d层中"""
    conv = copy.deepcopy(conv)
    conv.weight.requires_grad_(False)
    conv.bias.requires_grad_(False)

    if conv.bias is None:
        conv.bias = torch.zeros(
            conv.out_channels, device=conv.weight.device, dtype=conv.weight.dtype
        )

    bn_weight = bn.weight.data
    bn_bias = bn.bias.data
    bn_running_mean = bn.running_mean
    bn_running_var = bn.running_var
    bn_eps = bn.eps

    std = torch.sqrt(bn_running_var + bn_eps)
    scale = bn_weight / std

    conv.weight.data = conv.weight.data * scale.view(-1, 1, 1, 1)
    conv.bias.data = (conv.bias.data - bn_running_mean) * scale + bn_bias

    return conv


def export_model(model: SimpleCNN):
    """将模型参数导出到C语言头文件中"""
    model.eval()
    model.to("cpu")
    conv_layers = [
        fuse_conv_bn(model.conv_1, model.bn_1),
        fuse_conv_bn(model.conv_2, model.bn_2),
        fuse_conv_bn(model.conv_3, model.bn_3),
        model.conv_4,  # 最后一层没有BN
    ]

    # 自动生成C语言头文件，包含卷积核参数、偏置、输入输出尺寸等
    with open("C/include/cnn_demo.h", "w+", encoding="utf-8") as f:
        f.write(HEAD_BLOCK + "\n")
        for i, conv_layer in enumerate(conv_layers):
            f.write(
                CONV_BLOCK.format(
                    i + 1,
                    conv_layer.weight.shape[1],
                    i + 1,
                    conv_layer.weight.shape[0],
                    i + 1,
                    i + 1,
                    i + 1,
                    ",".join([str(x) for x in conv_layer.weight.flatten().tolist()]),
                    i + 1,
                    i + 1,
                    ",".join([str(x) for x in conv_layer.bias.flatten().tolist()]),
                )
                + "\n"
            )
            print(
                f"第{i + 1}层卷积层参数: {conv_layer.weight.shape}, 偏置: {conv_layer.bias.shape}"
            )

    total_params = 0
    for conv_layer in conv_layers:
        total_params += conv_layer.weight.numel()
        total_params += conv_layer.bias.numel()
    print(f"总参数量: {total_params}")


if __name__ == "__main__":
    model = torch.load(
        "./checkpoints/best_model.pth", map_location="cpu", weights_only=False
    )
    export_model(model)

    # 加载一张示例图像，打印出32x32的图像数据与pytorch的输出，用于验证C语言模型的正确性
    test_image = pil_loader("./data/test/dog/blenheim_spaniel_s_000123.png")
    test_image = test_image.convert("L")
    test_image = test_image.resize((IMAGE_SIZE, IMAGE_SIZE))
    test_image = pil_to_tensor(test_image) / 255.0
    print("图像数据: ")
    print(",".join([str(x) for x in test_image.flatten().tolist()]))
    print("Pytorch 模型输出: ", model(test_image[None]).detach().numpy().tolist())
