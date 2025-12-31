# 方舟平台图片生成节点使用指南

## 简介

本模块实现了火山引擎方舟平台的图片生成 API 接口，支持 Seedream 系列模型（4.5、4.0、3.0-t2i）和 Seededit 3.0-i2i 模型。

## 节点说明

### 1. 方舟图片生成节点 (ArkImageGenerationNode)

完整的方舟平台图片生成节点，支持所有参数配置。

**支持的模型：**
- `doubao-seedream-4.5` - 最新版本，支持组图、多图融合等高级功能
- `doubao-seedream-4.0` - 支持组图功能
- `doubao-seedream-3.0-t2i` - 文生图专用
- `doubao-seededit-3.0-i2i` - 图生图专用

**主要参数：**
- `model`: 选择使用的模型
- `prompt`: 文本提示词（支持中英文）
- `size`: 图片尺寸，格式如 "2048x2048"、"2K"、"4K" 等
- `image`: 可选的输入图片（支持单图或多图批次，最多14张）
- `image_url`: 图片URL（支持多个URL，用逗号或换行分隔，最多14张）
- `seed`: 随机种子（仅 3.0 模型支持）
- `guidance_scale`: 文本权重（仅 3.0 模型支持）
- `sequential_image_generation`: 组图模式（auto/disabled，仅 4.5/4.0 支持）
- `max_images`: 最大生成图片数（组图模式下有效）
- `stream`: 是否开启流式输出（仅 4.5/4.0 支持）
- `watermark`: 是否添加水印
- `response_format`: 返回格式（url/b64_json）

**功能特点：**
- ✅ 支持文生图、图生图
- ✅ 支持组图生成（最多15张）
- ✅ 支持多图融合（最多14张参考图）
- ✅ 支持批次图像输入（自动处理多张图片）
- ✅ 支持流式输出
- ✅ 支持自定义尺寸

### 2. 方舟智能图片生成节点 (ArkImageGenerationSmartNode)

简化版节点，自动选择最佳模型和参数。

**生成模式：**
- `文生图`: 仅根据文本提示词生成图片
- `图生图`: 根据输入图片和文本提示词生成图片
- `组图生成`: 生成一组内容关联的图片

**主要参数：**
- `prompt`: 文本提示词
- `mode`: 生成模式（文生图/图生图/组图生成）
- `image/image_url`: 输入图片（图生图模式需要）
- `width/height`: 图片宽高
- `max_images`: 最大生成图片数（组图模式）

**功能特点：**
- ✅ 自动选择最佳模型（默认使用 doubao-seedream-4.5）
- ✅ 简化参数配置
- ✅ 适合快速使用

## 使用示例

### 示例1: 文生图

```
节点: 方舟智能图片生成节点
- mode: 文生图
- prompt: "一只可爱的橘猫，坐在窗台上看着外面的风景，阳光洒在它身上"
- width: 2048
- height: 2048
```

### 示例2: 图生图

```
节点: 方舟智能图片生成节点
- mode: 图生图
- prompt: "将背景换成演唱会现场，增加舞台灯光效果"
- image: 连接输入图片
- width: 2048
- height: 2048
```

### 示例3: 组图生成

```
节点: 方舟智能图片生成节点
- mode: 组图生成
- prompt: "一只小猫的一天，从早晨醒来到晚上睡觉"
- max_images: 10
- width: 2048
- height: 2048
```

### 示例4: 多图融合生成

```
节点: 方舟图片生成节点
- model: doubao-seedream-4.5
- prompt: "融合这些图片的风格，创作一幅新的艺术作品"
- image: 连接多张图片（批次输入）或使用 image_url
- image_url: "url1, url2, url3" (多个URL用逗号分隔)
- size: 2048x2048
```

### 示例5: 高级参数控制

```
节点: 方舟图片生成节点
- model: doubao-seedream-4.5
- prompt: "科幻风格的未来城市"
- size: 4K
- sequential_image_generation: auto
- max_images: 15
- stream: true
- watermark: true
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

1. **API Key 获取**: 在[方舟平台控制台](https://console.volcengine.com/ark/region:ark+cn-beijing/apiKey)获取 API Key

2. **图片限制**:
   - 图片格式: jpeg、png、webp、bmp、tiff、gif (4.5/4.0)
   - 宽高比范围: [1/16, 16] (4.5/4.0) 或 [1/3, 3] (3.0)
   - 大小限制: 单张不超过 10MB
   - 总像素: 不超过 6000×6000 px
   - **多图输入**: Seedream 4.5/4.0 支持 2-14 张参考图

3. **尺寸设置**:
   - Seedream 4.5: 支持 2K、4K 或自定义宽高（总像素 [3686400, 16777216]）
   - Seedream 4.0: 支持 1K、2K、4K 或自定义宽高（总像素 [921600, 16777216]）
   - Seedream 3.0: 自定义宽高（总像素 [512x512, 2048x2048]）

4. **多图融合功能**:
   - 仅 Seedream 4.5 和 4.0 支持
   - 支持 2-14 张参考图输入
   - 可以使用 ComfyUI 批次图像输入
   - 可以使用多个 URL（用逗号或换行分隔）
   - 输入参考图数量 + 生成图片数量 ≤ 15 张

5. **组图功能**:
   - 仅 Seedream 4.5 和 4.0 支持
   - 输入参考图数量 + 生成图片数量 ≤ 15 张
   - 模型会自动判断生成的图片数量

6. **流式输出**:
   - 仅 Seedream 4.5 和 4.0 支持
   - 可以实时看到每张图片的生成进度

7. **计费方式**: 按成功生成的图片张数计费，生成失败的图片不计费

## 常见问题

**Q: 如何生成多张图片？**
A: 使用"组图生成"模式，或在高级节点中设置 `sequential_image_generation: auto`

**Q: 如何使用多张图片作为输入？**
A: 有以下几种方式：
1. **ComfyUI 批次图像**：使用 ComfyUI 的批次处理节点（如 Load Image Batch）生成多张图片，直接连接到 `image` 输入端口
2. **多个 URL**：在 `image_url` 参数中输入多个 URL，用逗号或换行分隔，例如：
   ```
   https://example.com/image1.jpg,
   https://example.com/image2.jpg,
   https://example.com/image3.jpg
   ```
3. **混合使用**：可以同时使用 `image` 和 `image_url`，节点会合并处理（最多14张）

注意：
- Seedream 4.5/4.0 支持 2-14 张参考图
- Seededit 3.0-i2i 只支持单图
- 超过 14 张时会自动使用前 14 张

**Q: 图片生成失败怎么办？**
A: 检查以下几点：
- API Key 是否正确
- 提示词是否符合规范（不超过300个汉字或600个英文单词）
- 图片尺寸是否在支持范围内
- 网络连接是否正常

**Q: 如何优化生成质量？**
A: 
- 使用详细、具体的提示词
- 参考[提示词指南](https://www.volcengine.com/docs/82379/1829186)
- 调整 `guidance_scale` 参数（仅 3.0 模型）
- 尝试不同的 seed 值

## 相关链接

- [方舟平台控制台](https://console.volcengine.com/ark)
- [API 文档](https://www.volcengine.com/docs/82379/1666945)
- [模型列表](https://www.volcengine.com/docs/82379/1330310)
- [计费说明](https://www.volcengine.com/docs/82379/1544106)
