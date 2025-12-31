# 方舟平台视频生成节点使用指南

## 简介

本模块实现了火山引擎方舟平台的视频生成 API 接口，支持 Seedance 系列模型（1.5 pro、1.0 pro、1.0 lite）。所有视频生成都是异步操作，需要通过轮询获取结果。

## 节点说明

### 1. 方舟视频生成节点 (ArkVideoGenerationNode)

完整的方舟平台视频生成节点，支持所有参数配置。

**支持的模型：**
- `doubao-seedance-1-5-pro-250228` - 最新版本，支持有声视频
- `doubao-seedance-1-0-pro-250228` - 高质量版本
- `doubao-seedance-1-0-pro-fast-250228` - 快速版本
- `doubao-seedance-1-0-lite-t2v-250228` - 轻量文生视频
- `doubao-seedance-1-0-lite-i2v-250228` - 轻量图生视频

**支持的生成模式：**
- **文生视频 (T2V)**: 仅输入文本提示词
- **首帧生成 (I2V-首帧)**: 根据首帧图片和文本生成视频
- **首尾帧生成 (I2V-首尾帧)**: 根据首尾帧图片和文本生成视频（仅支持 1.5 pro、1.0 pro）
- **参考图生成**: 使用 1-4 张参考图片生成视频（仅 1.0 lite-i2v）

**主要参数：**
- `model`: 选择使用的模型
- `prompt`: 视频描述文本（支持中英文，最长 500 字）
- `duration`: 视频时长（2-12 秒，1.5 pro 支持 4-12 秒或自动 -1）
- `resolution`: 分辨率（480p、720p、1080p）
- `ratio`: 宽高比（16:9、4:3、1:1、3:4、9:16、21:9、adaptive）
- `image_first/image_url_first`: 首帧图片（可选）
- `image_last/image_url_last`: 尾帧图片（可选）
- `seed`: 随机种子
- `watermark`: 是否添加水印
- `camera_fixed`: 是否固定摄像头
- `generate_audio`: 是否生成音频（仅 1.5 pro）
- `service_tier`: 服务等级（default/flex）
- `timeout`: 等待超时时间（秒）
- `poll_interval`: 轮询间隔（秒）

**功能特点：**
- ✅ 支持文生视频、首帧生成、首尾帧生成
- ✅ 支持有声视频生成（1.5 pro）
- ✅ 完全异步，自动轮询获取结果
- ✅ 自动提取视频第一帧作为预览
- ✅ 支持多种分辨率和宽高比

### 2. 方舟智能视频生成节点 (ArkVideoGenerationSmartNode)

简化版节点，自动判断生成模式。

**功能特点：**
- ✅ 自动判断生成模式（文生视频/首帧生成/首尾帧生成）
- ✅ 自动选择最新模型（Seedance 1.5 pro）
- ✅ 简化参数配置
- ✅ 适合快速使用

## 生成模式说明

### 文生视频（T2V）
**条件**: 不提供任何图片输入

```
节点: 方舟智能视频生成节点
- prompt: "一只小猫在阳光下打哈欠"
- duration: 5
- resolution: 720p
```

**输出**: 根据文本描述生成 5 秒视频

### 首帧生成（I2V-首帧）
**条件**: 提供首帧图片（image_first 或 image_url_first），不提供尾帧

```
节点: 方舟智能视频生成节点
- prompt: "小猫站起来，开始走动"
- image_first: 连接首帧图片
- duration: 5
- resolution: 720p
```

**输出**: 从给定的首帧开始，生成 5 秒视频

### 首尾帧生成（I2V-首尾帧）
**条件**: 同时提供首帧和尾帧图片

```
节点: 方舟智能视频生成节点
- prompt: "小猫从睡觉到醒来，看向镜头"
- image_first: 连接首帧图片（小猫睡觉）
- image_last: 连接尾帧图片（小猫醒来）
- duration: 8
- resolution: 720p
```

**输出**: 从首帧过渡到尾帧，生成 8 秒视频

## 使用示例

### 示例1: 快速文生视频

```
节点: 方舟智能视频生成节点
- prompt: "一群小鸟在林间飞翔"
- duration: 5
- resolution: 720p
- ratio: 16:9
```

### 示例2: 首帧生成视频

```
节点: 方舟视频生成节点
- model: doubao-seedance-1-5-pro-250228
- prompt: "镜头缓缓推进，展现整个场景"
- image_first: [Load Image] 输出
- duration: 6
- resolution: 720p
- ratio: adaptive
```

### 示例3: 首尾帧生成视频

```
节点: 方舟视频生成节点
- model: doubao-seedance-1-5-pro-250228
- prompt: "平稳的过渡，人物从远处走到近处"
- image_first: [Load Image] 远景
- image_last: [Load Image] 近景
- duration: 8
- resolution: 720p
- generate_audio: true
```

### 示例4: 高速生成（flex 服务等级）

```
节点: 方舟智能视频生成节点
- prompt: "快速变化的场景转换"
- duration: 4
- resolution: 480p
- service_tier: flex
- poll_interval: 10
```

## API Key 配置

有两种方式配置 API Key：

### 方式1: 在节点中直接输入
在节点的 `api_key` 参数中输入方舟平台的 API Key

### 方式2: 使用环境变量
设置环境变量 `ARK_API_KEY`：

**Windows (PowerShell):**
```powershell
$env:ARK_API_KEY = "your-api-key-here"
```

**Linux/Mac:**
```bash
export ARK_API_KEY="your-api-key-here"
```

## 注意事项

### 1. 图片限制
- 图片格式: jpeg、png、webp、bmp、tiff、gif（1.5 pro 新增支持 heic、heif）
- 宽高比（宽/高）: (0.4, 2.5)
- 宽高长度（px）: (300, 6000)
- 大小: 小于 30 MB

### 2. 视频规格

#### 不同模型对应的输出分辨率

| 分辨率 | 宽高比 | Seedance 1.0 系列 (px) | Seedance 1.5 pro (px) |
|--------|--------|----------------------|----------------------|
| 480p   | 16:9   | 864×480             | 864×496             |
|        | 4:3    | 736×544             | 752×560             |
|        | 1:1    | 640×640             | 640×640             |
|        | 3:4    | 544×736             | 560×752             |
|        | 9:16   | 480×864             | 496×864             |
|        | 21:9   | 960×416             | 992×432             |
| 720p   | 16:9   | 1248×704            | 1280×720            |
|        | 4:3    | 1120×832            | 1112×834            |
|        | 1:1    | 960×960             | 960×960             |
|        | 3:4    | 832×1120            | 834×1112            |
|        | 9:16   | 704×1248            | 720×1280            |
|        | 21:9   | 1504×640            | 1470×630            |
| 1080p  | 16:9   | 1920×1088           | -                   |
|        | 4:3    | 1664×1248           | -                   |
|        | 1:1    | 1440×1440           | -                   |
|        | 3:4    | 1248×1664           | -                   |
|        | 9:16   | 1088×1920           | -                   |
|        | 21:9   | 2176×928            | -                   |

#### 时长设置

- **Seedance 1.5 pro**: 支持 4-12 秒或设置为 -1（自动选择）
- **Seedance 1.0 系列**: 支持 2-12 秒

### 3. 首尾帧生成

- 首尾帧图片可以相同
- 首尾帧宽高比不一致时，以首帧为准，尾帧会自动裁剪适配
- 仅支持 Seedance 1.5 pro、1.0 pro、1.0 lite i2v 模型

### 4. 轮询设置

- `timeout`: 总等待时间（默认 300 秒）
- `poll_interval`: 每次轮询间隔（默认 5 秒）
- 更长的 timeout 适合生成较长视频或使用 flex 服务等级

### 5. 音频生成（1.5 pro）

- `generate_audio: true`: 生成同步音频（基于文本提示和视觉内容）
- `generate_audio: false`: 生成无声视频
- 建议将对话部分用双引号标注，以优化音频效果

### 6. 服务等级

- **default**: 在线推理模式，时效性更高，RPM 和并发配额较低
- **flex**: 离线推理模式，配额更高，价格为在线推理的 50%，适合批量生成

### 7. 计费方式

按照生成视频的总时长计费，具体详见[计费文档](https://www.volcengine.com/docs/82379/1544106)

## 输出说明

节点返回三个输出：

1. **image**: 视频的第一帧（PNG，用作预览）
2. **info**: 生成信息字符串，包含生成模式、分辨率、宽高比等
3. **video_url**: 视频下载链接，可用于下载完整视频

## 常见问题

**Q: 如何下载生成的视频？**
A: 使用节点返回的 `video_url` 参数，可以直接下载视频文件。视频 URL 在一定时间内有效。

**Q: 如何生成有声视频？**
A: 使用 Seedance 1.5 pro 模型，设置 `generate_audio: true`，将对话内容用双引号括起来。

**Q: 首尾帧生成的视频如何确保平稳过渡？**
A: 
- 使用内容相关的首尾帧
- 编写清晰的过渡描述（如"平稳推动摄像头"）
- 使用 adaptive 宽高比让模型自动适配

**Q: 轮询超时怎么办？**
A: 
- 增加 `timeout` 参数值
- 检查网络连接
- 使用 flex 服务等级可能提高成功率
- 查看日志了解具体错误信息

**Q: 如何优化视频生成质量？**
A:
- 使用详细、具体的文本描述
- 参考[提示词指南](https://www.volcengine.com/docs/82379/1587797)
- 使用合适的首帧和尾帧图片
- 尝试不同的 seed 值
- Seedance 1.5 pro 通常质量更高

## 依赖要求

- **可选**: 安装 ffmpeg 以提取视频第一帧预览
  ```bash
  # Windows (使用 Chocolatey)
  choco install ffmpeg
  
  # Linux
  sudo apt-get install ffmpeg
  
  # Mac
  brew install ffmpeg
  ```

## 相关链接

- [方舟平台控制台](https://console.volcengine.com/ark)
- [视频生成 API 文档](https://www.volcengine.com/docs/82379/1520758)
- [模型列表](https://www.volcengine.com/docs/82379/1330310)
- [提示词指南](https://www.volcengine.com/docs/82379/1587797)
- [计费说明](https://www.volcengine.com/docs/82379/1544106)
