# CNN Demo

一个极简的卷积神经网络演示项目，支持训练和导出为C语言代码。

## 环境准备

### 1. 安装 uv

uv 是一个快速的 Python 包管理器和项目管理工具。

**Windows:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Linux/macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

更多安装方式请参考 [uv 官方文档](https://github.com/astral-sh/uv)。

### 2. 同步项目依赖

在项目根目录下运行：

```bash
uv sync
```

这将根据 `pyproject.toml` 安装所有必需的依赖包。

## 数据集准备

### 训练数据集目录组织方式

项目使用标准的图像分类数据集目录结构：

```
data/
├── train/
│   ├── cat/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── dog/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── test/
    ├── cat/
    │   ├── image1.jpg
    │   └── ...
    └── dog/
        ├── image1.jpg
        └── ...
```

- `data/train/` 目录存放训练集，每个类别一个子文件夹
- `data/test/` 目录存放测试集，每个类别一个子文件夹
- 图像会自动调整为 32x32 灰度图像

## 使用说明

### 启动训练

使用 uv 运行训练脚本：

```bash
uv run main.py
```

训练过程会：
- 自动加载数据集
- 训练模型并保存最佳检查点到 `checkpoints/best_model.pth`
- 显示训练和测试的损失值及准确率

### 启动导出

训练完成后，使用以下命令导出模型为C语言头文件：

```bash
uv run export.py
```

导出脚本会：
- 加载最佳模型检查点
- 将 BatchNorm 参数融合到卷积层中
- 生成 C 语言头文件 `C/include/cnn_demo.h`，包含所有模型参数
- 打印示例图像的像素数据和模型输出，用于验证C语言实现的正确性

## C语言部署

### 导出文件说明

导出过程会生成 `C/include/cnn_demo.h` 头文件，包含：
- 图像尺寸定义（IMAGE_H, IMAGE_W）
- 卷积核尺寸定义（KERNEL_H, KERNEL_W）
- 各层卷积的权重和偏置参数

### 编译和运行

#### 方式一：使用 CMake

项目已包含 `C/CMakeLists.txt`，可以使用 CMake 构建：

```bash
cd C
mkdir build
cd build
cmake ..
cmake --build .
```

#### 方式二：复制源码到 IDE

也可以直接将 `C/cnn_demo.c` 和 `C/include/cnn_demo.h` 复制到你的 IDE 项目中编译运行。

## 算法说明

### 当前局限性

1. **网络规模较小**：网络层数和通道数都很小，参数量共约 26K，表达能力有限。

2. **算子种类单一**：在 32 分辨率下只含有一个固定步长、核大小、填充方式的卷积层算子（3x3 卷积，步长为 2，padding 为 1）。不含全连接层。

3. **优缺点**：
   - **优点**：易于移植实现，运算速度快，适合嵌入式设备
   - **缺点**：准确率较低，参考意义更大

### 进一步改进方向

1. **增大参数量**：增加网络层数和通道数，提升模型表达能力。

2. **增加算子种类**：
   - 3x3 卷积，步长为 1
   - 1x1 卷积，步长为 1
   - 逐元素求和（残差连接）

3. **优化训练策略**：
   - 超参数调优
   - 损失函数优化
   - 标签平滑（Label Smoothing）
   - 知识蒸馏（Knowledge Distillation）

4. **量化优化**：
   - 训练阶段进行 int8 量化感知训练
   - 部署阶段进行 int8 量化，减少模型大小和推理时间

