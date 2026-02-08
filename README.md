# Qwen3 Forced Aligner

[![Build and Release](https://github.com/WEIFENG2333/qwen3-forced-aligner/actions/workflows/build-release.yml/badge.svg)](https://github.com/WEIFENG2333/qwen3-forced-aligner/actions/workflows/build-release.yml)
[![Test](https://github.com/WEIFENG2333/qwen3-forced-aligner/actions/workflows/test.yml/badge.svg)](https://github.com/WEIFENG2333/qwen3-forced-aligner/actions/workflows/test.yml)

基于 [Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) 的音频-文本强制对齐工具。给定一段音频和对应文本，输出每个词/字的时间戳。

支持 CLI 命令行和 REST API 两种使用方式，支持 11 种语言。提供 Linux、macOS、Windows 预编译包，开箱即用。

## 安装

### 方式一：下载预编译包

从 [Releases](https://github.com/WEIFENG2333/qwen3-forced-aligner/releases) 下载对应平台的压缩包，解压后即可使用。模型已内置，无需额外下载。

```bash
./qwen3-aligner align -a audio.wav -t "你好世界" -l zh
```

### 方式二：从源码安装

```bash
git clone https://github.com/WEIFENG2333/qwen3-forced-aligner.git
cd qwen3-forced-aligner

# 推荐使用 uv（默认安装 CPU 版 PyTorch）
uv sync
qwen3-aligner download-model
qwen3-aligner align -a audio.wav -t "Hello world" -l en
```

> 项目已配置 `tool.uv.sources` 从 PyTorch CPU 索引安装 torch，无需手动处理 CUDA 依赖。如需 GPU 推理，可自行替换为 CUDA 版本：`uv pip install torch --index-url https://download.pytorch.org/whl/cu124 --reinstall`

## 使用

### 命令行

```bash
# 中文对齐
qwen3-aligner align -a audio.wav -t "欢迎使用语音对齐" -l zh

# 英文对齐，输出 JSON
qwen3-aligner align -a audio.wav -t "Hello world" -l en -f json

# 日文对齐，结果写入文件
qwen3-aligner align -a audio.wav -t "こんにちは" -l ja -o result.json -f json

# 指定模型路径
qwen3-aligner align -a audio.wav -t "text" -l zh -m ./my-models

# CUDA 推理（需要 GPU）
qwen3-aligner align -a audio.wav -t "text" -l zh --dtype bfloat16
```

### REST API 服务

```bash
# 启动服务
qwen3-aligner serve -p 8765

# 设置模型保活时间（秒），-1 表示永不卸载
qwen3-aligner serve -p 8765 -k 300
```

请求示例：

```bash
curl -X POST http://localhost:8765/align \
  -H "Content-Type: application/json" \
  -d '{"audio": "/path/to/audio.wav", "text": "你好世界", "language": "zh"}'
```

返回格式：

```json
{
  "success": true,
  "alignments": [
    {"text": "你", "start_time": 0.24, "end_time": 0.44},
    {"text": "好", "start_time": 0.44, "end_time": 0.64},
    {"text": "世", "start_time": 0.64, "end_time": 0.80},
    {"text": "界", "start_time": 0.80, "end_time": 0.96}
  ],
  "processing_time": 0.25
}
```

其他接口：

| 接口 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/model/status` | GET | 模型状态 |
| `/model/load` | POST | 预加载模型 |
| `/model/unload` | POST | 卸载模型释放内存 |
| `/config` | GET | 查看当前配置 |

### 模型管理

```bash
# 下载模型到默认位置
qwen3-aligner download-model

# 下载到指定目录
qwen3-aligner download-model -o ./my-models

# 查看模型信息
qwen3-aligner model-info
```

## 支持的语言

| 语言 | 代码 | 语言 | 代码 |
|------|------|------|------|
| Chinese 中文 | `zh` | Italian 意大利语 | `it` |
| Cantonese 粤语 | `yue` | Portuguese 葡萄牙语 | `pt` |
| English 英语 | `en` | Russian 俄语 | `ru` |
| German 德语 | `de` | Korean 韩语 | `ko` |
| Spanish 西班牙语 | `es` | Japanese 日语 | `ja` |
| French 法语 | `fr` | | |

`-l` 参数支持全称（如 `Chinese`）和缩写（如 `zh`），不区分大小写。

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `ALIGNER_MODEL_PATH` | 模型路径 | 自动检测 |
| `ALIGNER_DEVICE` | 设备 (cpu/cuda) | cpu |
| `ALIGNER_DTYPE` | 数据类型 (float32/bfloat16) | float32 |
| `ALIGNER_HOST` | 服务绑定地址 | 0.0.0.0 |
| `ALIGNER_PORT` | 服务端口 | 8765 |
| `ALIGNER_KEEP_ALIVE` | 模型保活时间（秒） | 300 |

## 从源码构建

```bash
# 安装开发依赖
uv sync --extra dev

# 构建
python build.py

# 构建并打包为 zip（模型需自行复制到 dist 目录）
python build.py --archive
```

构建产物在 `dist/qwen3-aligner/` 目录下。

## License

Apache 2.0

## 致谢

- [Qwen3-ForcedAligner](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) - Alibaba Qwen Team
- [qwen_asr](https://github.com/QwenLM/Qwen3-ASR) - Qwen3 ASR 核心库
