"""
火山引擎方舟平台图片生成 API 节点
支持 Seedream 4.5、4.0、3.0-t2i 和 Seededit 3.0-i2i 等模型
"""

import base64
import io
import os
import requests
import json
from PIL import Image
import numpy as np
import torch
import time


class ArkImageGenerationNode:
    """方舟平台图片生成节点 - 支持 Seedream 系列模型"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["doubao-seedream-4-5-251128", "doubao-seedream-4-0-250828", "doubao-seedream-3-0-t2i-250415", "doubao-seededit-3-0-i2i-250628"],),
                "prompt": ("STRING", {"default": "一只可爱的小猫", "multiline": True}),
                "size": ("STRING", {"default": "2048x2048", "multiline": False}),
            },
            "optional": {
                "image": ("IMAGE",),  # 可选的输入图片
                "image_url": ("STRING", {"default": "", "placeholder": "图片URL（用于图生图）"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "guidance_scale": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 10.0, "step": 0.1}),
                "response_format": (["url", "b64_json"],),
                "watermark": ("BOOLEAN", {"default": False}),
                "sequential_image_generation": (["auto", "disabled"],),
                "max_images": ("INT", {"default": 15, "min": 1, "max": 15}),
                "stream": ("BOOLEAN", {"default": False}),
                "api_key": ("STRING", {"default": "", "placeholder": "方舟平台 API Key"}),
                "timeout": ("INT", {"default": 120, "min": 30, "max": 600, "step": 10}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "generate_image"
    CATEGORY = "Volcano/Ark"
    
    def generate_image(self, model, prompt, size, image=None, image_url="", seed=-1, 
                      guidance_scale=2.5, response_format="url", watermark=False,
                      sequential_image_generation="disabled", max_images=15, stream=False,
                      api_key="", timeout=120):
        try:
            # 获取 API Key
            if not api_key or not api_key.strip():
                api_key = os.getenv('ARK_API_KEY')
                if not api_key:
                    raise Exception("未设置方舟平台 API Key，请在节点参数或环境变量 ARK_API_KEY 中设置")
            
            # API 端点
            api_url = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
            
            # 构建请求头
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # 构建请求体
            request_body = {
                "model": model,
                "prompt": prompt,
                "size": size,
                "response_format": response_format,
                "watermark": watermark
            }
            
            # 处理图片输入（用于图生图）
            image_inputs = []
            if image is not None:
                # 处理 ComfyUI 图像张量
                if isinstance(image, torch.Tensor):
                    # 检查是否是批次图像
                    if len(image.shape) == 4:
                        # 批次图像，处理每一张
                        batch_size = image.shape[0]
                        print(f"检测到批次图像，共 {batch_size} 张")
                        
                        # 限制最多处理 14 张图片
                        max_process = min(batch_size, 14)
                        if batch_size > 14:
                            print(f"警告：输入了 {batch_size} 张图片，但最多只支持 14 张，将只使用前 14 张")
                        
                        for i in range(max_process):
                            input_image = image[i]
                            image_array = input_image.cpu().numpy()
                            image_array = (image_array * 255).astype(np.uint8)
                            pil_image = Image.fromarray(image_array)
                            
                            if pil_image.mode != 'RGB':
                                pil_image = pil_image.convert('RGB')
                            
                            # 转换为 base64
                            buffer = io.BytesIO()
                            pil_image.save(buffer, format='PNG')
                            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                            image_inputs.append(f"data:image/png;base64,{image_base64}")
                    else:
                        # 单张图像
                        image_array = image.cpu().numpy()
                        image_array = (image_array * 255).astype(np.uint8)
                        pil_image = Image.fromarray(image_array)
                        
                        if pil_image.mode != 'RGB':
                            pil_image = pil_image.convert('RGB')
                        
                        # 转换为 base64
                        buffer = io.BytesIO()
                        pil_image.save(buffer, format='PNG')
                        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                        image_inputs.append(f"data:image/png;base64,{image_base64}")
            
            # 处理 image_url 参数（支持多个URL，用逗号或换行分隔）
            if image_url and image_url.strip():
                # 支持多种分隔符：逗号、换行符
                urls = [url.strip() for url in image_url.replace('\n', ',').split(',') if url.strip()]
                for url in urls:
                    if len(image_inputs) >= 14:
                        print(f"警告：已达到最大图片数量限制（14张），忽略剩余URL")
                        break
                    image_inputs.append(url)
            
            # 如果有图片输入，添加到请求体
            if image_inputs:
                print(f"共处理 {len(image_inputs)} 张输入图片")
                
                # doubao-seededit-3-0-i2i-250628 只支持单图
                if model == "doubao-seededit-3-0-i2i-250628":
                    if len(image_inputs) > 1:
                        print(f"警告：{model} 只支持单图输入，将只使用第一张图片")
                    request_body["image"] = image_inputs[0]
                # doubao-seedream 系列支持多图
                elif model in ["doubao-seedream-4-5-251128", "doubao-seedream-4-0-250828"]:
                    # 如果只有一张图，直接传字符串；多张图传数组
                    if len(image_inputs) == 1:
                        request_body["image"] = image_inputs[0]
                    else:
                        request_body["image"] = image_inputs
            
            # 添加 seed（仅 3.0 模型支持）
            if model in ["doubao-seedream-3-0-t2i-250415", "doubao-seededit-3-0-i2i-250628"]:
                if seed != -1:
                    request_body["seed"] = seed
                # guidance_scale
                if model == "doubao-seedream-3-0-t2i-250415":
                    request_body["guidance_scale"] = guidance_scale
                elif model == "doubao-seededit-3-0-i2i-250628":
                    request_body["guidance_scale"] = guidance_scale if guidance_scale != 2.5 else 5.5
            
            # 添加组图相关参数（仅 4.5 和 4.0 支持）
            if model in ["doubao-seedream-4-5-251128", "doubao-seedream-4-0-250828"]:
                request_body["sequential_image_generation"] = sequential_image_generation
                if sequential_image_generation == "auto":
                    request_body["sequential_image_generation_options"] = {
                        "max_images": max_images
                    }
                # 流式输出
                request_body["stream"] = stream
            
            # 发送请求
            print(f"开始调用方舟平台图片生成 API，模型: {model}")
            
            if stream and model in ["doubao-seedream-4-5-251128", "doubao-seedream-4-0-250828"]:
                # 流式请求
                return self._handle_stream_response(api_url, headers, request_body, timeout)
            else:
                # 非流式请求
                response = requests.post(api_url, headers=headers, json=request_body, timeout=timeout)
                
                if response.status_code != 200:
                    error_msg = f"API 请求失败: {response.status_code} - {response.text}"
                    raise Exception(error_msg)
                
                result = response.json()
                return self._process_response(result, response_format, timeout)
                
        except Exception as e:
            error_msg = f"图片生成失败: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)
    
    def _handle_stream_response(self, api_url, headers, request_body, timeout):
        """处理流式响应"""
        try:
            all_images = []
            
            with requests.post(api_url, headers=headers, json=request_body, stream=True, timeout=timeout) as response:
                if response.status_code != 200:
                    raise Exception(f"流式请求失败: {response.status_code} - {response.text}")
                
                # 逐行读取 SSE 数据
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            data_str = line_str[6:]  # 去掉 "data: " 前缀
                            if data_str.strip() == '[DONE]':
                                break
                            
                            try:
                                data = json.loads(data_str)
                                # 处理每个事件
                                if 'data' in data:
                                    for item in data['data']:
                                        if 'url' in item or 'b64_json' in item:
                                            # 下载或解码图片
                                            img_array = self._download_or_decode_image(item, timeout)
                                            if img_array is not None:
                                                all_images.append(img_array)
                                                print(f"已接收第 {len(all_images)} 张图片")
                            except json.JSONDecodeError:
                                continue
            
            if not all_images:
                raise Exception("流式响应中未收到任何图片")
            
            # 合并所有图片
            batch_images = np.stack(all_images, axis=0)
            image_tensor = torch.from_numpy(batch_images)
            info = f"生成成功！共 {len(all_images)} 张图片（流式输出）"
            
            return (image_tensor, info)
            
        except Exception as e:
            raise Exception(f"流式响应处理失败: {str(e)}")
    
    def _process_response(self, result, response_format, timeout):
        """处理非流式响应"""
        try:
            # 检查是否有错误
            if 'error' in result:
                error_info = result['error']
                raise Exception(f"API 错误 [{error_info.get('code', 'unknown')}]: {error_info.get('message', 'unknown error')}")
            
            # 获取图片数据
            if 'data' not in result or not result['data']:
                raise Exception("响应中未找到图片数据")
            
            all_images = []
            success_count = 0
            
            for idx, item in enumerate(result['data']):
                # 检查单张图片是否有错误
                if 'error' in item:
                    error_info = item['error']
                    print(f"第 {idx + 1} 张图片生成失败: [{error_info.get('code', 'unknown')}] {error_info.get('message', 'unknown error')}")
                    continue
                
                # 下载或解码图片
                img_array = self._download_or_decode_image(item, timeout)
                if img_array is not None:
                    all_images.append(img_array)
                    success_count += 1
            
            if not all_images:
                raise Exception("没有成功生成任何图片")
            
            # 合并所有图片
            batch_images = np.stack(all_images, axis=0)
            image_tensor = torch.from_numpy(batch_images)
            
            # 构建信息
            usage = result.get('usage', {})
            info = f"生成成功！共 {success_count} 张图片"
            if 'generated_images' in usage:
                info += f"，计费 {usage['generated_images']} 张"
            if 'size' in result['data'][0]:
                info += f"，尺寸: {result['data'][0]['size']}"
            
            return (image_tensor, info)
            
        except Exception as e:
            raise Exception(f"响应处理失败: {str(e)}")
    
    def _download_or_decode_image(self, item, timeout):
        """下载或解码单张图片"""
        try:
            if 'url' in item:
                # 从 URL 下载
                url = item['url']
                response = requests.get(url, timeout=timeout)
                if response.status_code == 200:
                    image_data = response.content
                else:
                    print(f"下载图片失败: {response.status_code}")
                    return None
            elif 'b64_json' in item:
                # 从 base64 解码
                image_data = base64.b64decode(item['b64_json'])
            else:
                print("图片数据中既没有 url 也没有 b64_json")
                return None
            
            # 创建图片
            pil_image = Image.open(io.BytesIO(image_data))
            
            # 转换为 RGB
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # 转换为 ComfyUI 格式
            image_array = np.array(pil_image).astype(np.float32) / 255.0
            
            return image_array
            
        except Exception as e:
            print(f"处理图片数据失败: {str(e)}")
            return None


class ArkImageGenerationSmartNode:
    """方舟平台智能图片生成节点 - 自动判断生成模式"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "一只可爱的小猫", "multiline": True}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image_url": ("STRING", {"default": "", "placeholder": "图片URL（可选，用于图生图/多图融合）", "multiline": True}),
                "width": ("INT", {"default": 2048, "min": 512, "max": 4096, "step": 64}),
                "height": ("INT", {"default": 2048, "min": 512, "max": 4096, "step": 64}),
                "max_images": ("INT", {"default": 1, "min": 1, "max": 15}),
                "watermark": ("BOOLEAN", {"default": False}),
                "api_key": ("STRING", {"default": "", "placeholder": "方舟平台 API Key"}),
                "timeout": ("INT", {"default": 120, "min": 30, "max": 600, "step": 10}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "smart_generate"
    CATEGORY = "Volcano/Ark"
    
    def smart_generate(self, prompt, image=None, image_url="", width=2048, height=2048, 
                      max_images=1, watermark=False, api_key="", timeout=120):
        try:
            # 智能判断生成模式
            has_image_input = (image is not None) or (image_url and image_url.strip())
            enable_sequential = max_images > 1
            
            # 自动选择模型（默认使用最新的 4.5）
            model = "doubao-seedream-4-5-251128"
            size = f"{width}x{height}"
            
            # 确定生成模式
            if has_image_input:
                # 检查是否是多图输入
                num_input_images = 0
                if image is not None and isinstance(image, torch.Tensor) and len(image.shape) == 4:
                    num_input_images = image.shape[0]
                elif image is not None:
                    num_input_images = 1
                
                if image_url and image_url.strip():
                    urls = [url.strip() for url in image_url.replace('\n', ',').split(',') if url.strip()]
                    num_input_images += len(urls)
                
                if num_input_images > 1:
                    mode_desc = f"多图融合（{num_input_images}张参考图）"
                else:
                    mode_desc = "图生图"
                
                print(f"[智能模式] 检测到图片输入，使用 {mode_desc}")
            else:
                mode_desc = "文生图"
                print(f"[智能模式] 未检测到图片输入，使用 {mode_desc}")
            
            # 根据 max_images 决定是否开启组图
            if enable_sequential:
                sequential_image_generation = "auto"
                print(f"[智能模式] 启用组图生成，最多生成 {max_images} 张")
            else:
                sequential_image_generation = "disabled"
            
            # 调用基础节点
            ark_node = ArkImageGenerationNode()
            result = ark_node.generate_image(
                model=model,
                prompt=prompt,
                size=size,
                image=image,
                image_url=image_url,
                seed=-1,
                guidance_scale=2.5,
                response_format="url",
                watermark=watermark,
                sequential_image_generation=sequential_image_generation,
                max_images=max_images,
                stream=False,
                api_key=api_key,
                timeout=timeout
            )
            
            # 在信息中添加模式标识
            images, info = result
            info = f"[智能-{mode_desc}] {info}"
            
            return (images, info)
            
        except Exception as e:
            error_msg = f"智能生成失败: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "ArkImageGenerationNode": ArkImageGenerationNode,
    "ArkImageGenerationSmartNode": ArkImageGenerationSmartNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkImageGenerationNode": "方舟图片生成 (Ark Image Generation)",
    "ArkImageGenerationSmartNode": "方舟智能图片生成 (Ark Smart Generation)",
}
