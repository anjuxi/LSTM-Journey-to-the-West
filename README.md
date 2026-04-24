# LSTM 中文文本生成 - 西游记

基于 LSTM（长短期记忆网络）的中文文本生成项目，使用《西游记》原著作为训练数据，学习古典文学的语言风格并生成类似风格的文本。

## 项目简介

本项目使用 PyTorch 框架构建了一个多层 LSTM 神经网络，通过字符级别的语言建模来学习《西游记》的文本特征，实现自动生成具有古典文学风格的中文文本。

## 项目结构
```
LSTM/
├── data/
│   └── xyj.txt   # 《西游记》原著文本数据
├── models/
│   ├── lstm_xiyouji.pth   # 训练好的模型权重
│   └── vocab.pkl    # 词汇表映射文件
├── lstm-text-generated.ipynb # 主程序（Jupyter Notebook）
└── README.md   # 项目说明文档
```
## 功能特性

- **字符级语言建模**：以单个汉字为最小单位进行建模，能够生成任意汉字组合
- **多层 LSTM 架构**：3层 LSTM 网络，增强模型的表达能力
- **Dropout 正则化**：防止过拟合，提高模型泛化能力
- **混合精度训练**：支持自动混合精度（AMP），加速训练并减少显存占用
- **断点续训**：支持从已保存的模型继续训练
- **温度采样**：可调节的 temperature 参数控制生成文本的随机性
- **多 GPU 支持**：支持 DataParallel 多卡并行训练

## 模型架构

| 组件 | 配置 |
|------|------|
| 词嵌入层 | Embedding(4198, 256) |
| LSTM 层 | 3层，隐藏维度 512 |
| Dropout | 0.3 |
| 输出层 | Linear(512, 4198) |

**模型参数量**：约 900 万参数（34.36 MB）

## 环境要求

- 相关环境依赖请进入jupyter notebook查看

## 使用方法

### 1. 训练模型

打开 `lstm-text-generated.ipynb`，按顺序运行所有单元格即可开始训练。

关键配置参数（可在 Notebook 中修改）：

```python
# 数据配置
SEQ_LENGTH = 100      # 输入序列长度
BATCH_SIZE = 1024     # 批次大小

# 模型配置
EMBED_DIM = 256       # 嵌入维度
HIDDEN_DIM = 512      # LSTM 隐藏层维度
NUM_LAYERS = 3        # LSTM 层数
DROPOUT = 0.3         # Dropout 概率

# 训练配置
EPOCHS = 10           # 训练轮数
LEARNING_RATE = 0.001 # 学习率
CONTINUE_TRAIN = 1    # 是否从已有模型继续训练

# 生成配置
GENERATE_LENGTH = 500 # 生成文本长度
TEMPERATURE = 0.1     # 采样温度（越低越保守，越高越随机）
```

### 2. 生成文本

训练完成后，使用以下方式生成文本：

```python
# 加载模型
model = load_model_for_inference(MODEL_SAVE_PATH, VOCAB_SAVE_PATH, device)

# 生成文本
generated_text = generate_text(model, start_text="话说", length=500, device=device, temperature=0.8)
print(generated_text)
```

### 3. 温度参数说明

- **temperature < 0.5**：生成文本更保守、确定性更强，接近原文风格
- **temperature ≈ 0.8**：平衡创造性和连贯性
- **temperature > 1.0**：生成文本更随机、创造性更强，但可能出现语法错误

## 训练数据

- **数据来源**：《西游记》原著
- **文本长度**：约 73.5 万字符
- **词汇表大小**：4198 个唯一字符

## 示例输出

使用不同的起始文本可以生成不同风格的段落：

| 起始文本 | 生成风格 |
|---------|---------|
| 话说 | 叙述性开场 |
| 悟空 | 孙悟空相关情节 |
| 唐僧 | 唐僧相关情节 |
| 那妖怪 | 战斗场景描述 |

## 技术细节

### 数据预处理
- 去除多余空白和特殊字符
- 构建字符到索引的映射表
- 将文本转换为整数序列

### 训练策略
- 使用 Adam 优化器
- 学习率调度（ReduceLROnPlateau）
- 梯度裁剪（防止梯度爆炸）
- 每 1 个 epoch 保存一次模型

### 生成策略
- 使用 softmax 概率分布采样
- 支持温度调节控制随机性
- 模型预热（priming）机制

## 注意事项

1. 建议使用 GPU 进行训练，CPU 训练速度较慢
2. 生成的文本可能存在语法不通顺的情况，可通过降低 temperature 改善

## 许可证

本项目仅供学习和研究使用。

## 致谢

- 训练数据来源于《西游记》原著
- 使用 PyTorch 深度学习框架

>作者：Ailan Anjuxi
>联系方式：
  邮箱：anjuxi.ME@outlook.com
  SIP电话：sip:anjuxi@sip.linphone.org