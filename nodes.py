

import base64
import io
import os
import requests
from volcengine.visual.VisualService import VisualService
from PIL import Image
import numpy as np
import torch


def create_visual_service_with_timeout(timeout=120):
    """创建带超时设置的VisualService实例"""
    visual_service = VisualService()
    
    # 尝试设置超时时间
    try:
        visual_service.set_connection_timeout(timeout)
        visual_service.set_socket_timeout(timeout)
            
    except Exception as e:
        print(f"警告：无法设置火山引擎SDK超时时间: {e}")
        
    return visual_service



class MaletteVolcanoT2IAPI:
    """火山引擎文本到图像生成节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "一只可爱的小猫", "multiline": True}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 4096, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 4096, "step": 64}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "req_key": ("STRING", {"default": "jimeng_t2i_v31", "options": ["jimeng_t2i_v31", "jimeng_t2i_v30"]}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "placeholder": "火山引擎 API Key"}),
                "secret_key": ("STRING", {"default": "", "placeholder": "火山引擎 Secret Key"}),
                "timeout": ("INT", {"default": 120, "min": 30, "max": 600, "step": 10}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "generate_image"
    CATEGORY = "image generation"
    
    def generate_image(self, prompt, width, height, seed, req_key, num_images, api_key="", secret_key="", timeout=120):
        try:
            # 初始化火山引擎服务
            visual_service = create_visual_service_with_timeout(timeout)

            # 设置 API 密钥（优先使用输入参数，否则使用环境变量或默认值）
            if api_key and secret_key:
                visual_service.set_ak(api_key)
                visual_service.set_sk(secret_key)
            else:
                # 尝试从环境变量获取
                env_ak = os.getenv('VOLC_AK')
                env_sk = os.getenv('VOLC_SK')
                if env_ak and env_sk:
                    visual_service.set_ak(env_ak)
                    visual_service.set_sk(env_sk)
                else:
                    raise Exception("火山引擎 API Key 或 Secret Key 未设置")
            
            # 并行生成多张图片
            all_images = []
            import concurrent.futures
            import time
            
            def generate_single_image(image_index):
                try:
                    # 为每个线程创建独立的 VisualService 实例
                    thread_visual_service = create_visual_service_with_timeout(timeout)
                    
                    # 设置 API 密钥
                    if api_key and secret_key:
                        thread_visual_service.set_ak(api_key)
                        thread_visual_service.set_sk(secret_key)
                    else:
                        # 尝试从环境变量获取
                        env_ak = os.getenv('VOLC_AK')
                        env_sk = os.getenv('VOLC_SK')
                        if env_ak and env_sk:
                            thread_visual_service.set_ak(env_ak)
                            thread_visual_service.set_sk(env_sk)
                        else:
                            raise Exception("火山引擎 API Key 或 Secret Key 未设置")
                    
                    # 为每张图片生成不同的种子
                    current_seed = seed if seed != -1 else int(time.time() * 1000) + image_index
                    if current_seed > 99999999:
                        current_seed = current_seed % 99999999
                    
                    # 构建请求参数
                    form = {
                        "req_key": req_key,
                        "prompt": prompt,
                        "seed": current_seed,
                        "width": width,
                        "height": height
                    }
                    
                    # 调用火山引擎 API
                    resp = thread_visual_service.cv_process(form)
                    
                    # 检查响应状态
                    if not resp or 'data' not in resp:
                        raise Exception(f"API 响应无效: {resp}")
                    
                    resp_data = resp['data']
                    
                    # 检查是否有错误
                    if 'error' in resp_data:
                        raise Exception(f"API 错误: {resp_data['error']}")
                    
                    # 获取图像数据
                    if 'binary_data_base64' not in resp_data or not resp_data['binary_data_base64']:
                        raise Exception("响应中未找到图像数据")
                    
                    binary_data_base64 = resp_data['binary_data_base64'][0]
                    
                    # 解码并创建图像
                    image_data = base64.b64decode(binary_data_base64)
                    pil_image = Image.open(io.BytesIO(image_data))
                    
                    # 转换为 RGB 模式（ComfyUI 需要）
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                    
                    # 转换为 ComfyUI 图像格式
                    image_array = np.array(pil_image).astype(np.float32) / 255.0
                    
                    return image_array
                    
                except Exception as e:
                    print(f"生成第 {image_index + 1} 张图片失败: {str(e)}")
                    # 返回错误图像
                    error_image = np.zeros((height, width, 3), dtype=np.float32)
                    error_image[:, :, 0] = 1.0  # 红色背景表示错误
                    return error_image
            
            # 使用线程池并行执行
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(num_images, 4)) as executor:
                # 提交所有任务
                future_to_index = {executor.submit(generate_single_image, i): i for i in range(num_images)}
                
                # 收集结果
                for future in concurrent.futures.as_completed(future_to_index):
                    image_array = future.result()
                    all_images.append(image_array)
            
            # 将所有图片合并为一个批次
            if all_images:
                # 确保所有图片都有相同的形状
                first_image = all_images[0]
                for i, img in enumerate(all_images):
                    if img.shape != first_image.shape:
                        # 如果形状不匹配，调整到第一张图片的形状
                        temp_img = Image.fromarray((img * 255).astype(np.uint8))
                        temp_img = temp_img.resize((first_image.shape[1], first_image.shape[0]))
                        all_images[i] = np.array(temp_img).astype(np.float32) / 255.0
                
                # 堆叠所有图片
                batch_images = np.stack(all_images, axis=0)
                
                # 转换为 PyTorch 张量（ComfyUI 需要）
                image_tensor = torch.from_numpy(batch_images)
                
                # 构建信息字符串
                info = f"生成成功！共 {len(all_images)} 张图片，尺寸: {width}x{height}, 种子: {seed}, 模型: {req_key}"
                
                return (image_tensor, info)
            else:
                raise Exception("没有成功生成任何图片")
            
        except Exception as e:
            error_msg = f"图像生成失败: {str(e)}"
            print(error_msg)
            
            raise Exception(error_msg)


class MaletteVolcanoI2IAPI:
    """火山引擎图像到图像生成节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "背景换成演唱会现场", "multiline": True}),
                "scale": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "req_key": ("STRING", {"default": "jimeng_i2i_v30", "options": ["jimeng_i2i_v30"]}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "placeholder": "火山引擎 API Key"}),
                "secret_key": ("STRING", {"default": "", "placeholder": "火山引擎 Secret Key"}),
                "timeout": ("INT", {"default": 120, "min": 30, "max": 600, "step": 10}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "generate_i2i"
    CATEGORY = "image generation"
    
    def generate_i2i(self, image, prompt, scale, seed, req_key, num_images, api_key="", secret_key="", timeout=120):
        try:
            # 初始化火山引擎服务
            visual_service = create_visual_service_with_timeout(timeout)
            
            # 设置 API 密钥（优先使用输入参数，否则使用环境变量或默认值）
            if api_key and secret_key:
                visual_service.set_ak(api_key)
                visual_service.set_sk(secret_key)
            else:
                # 尝试从环境变量获取
                env_ak = os.getenv('VOLC_AK')
                env_sk = os.getenv('VOLC_SK')
                if env_ak and env_sk:
                    visual_service.set_ak(env_ak)
                    visual_service.set_sk(env_sk)
                else:
                    raise Exception("火山引擎 API Key 或 Secret Key 未设置")
            
            # 将 ComfyUI 图像转换为 PIL Image
            if isinstance(image, torch.Tensor):
                # 如果是批次图像，取第一张
                if len(image.shape) == 4:
                    input_image = image[0]
                else:
                    input_image = image
                # 转换为 numpy 数组
                image_array = input_image.cpu().numpy()
                # 转换为 0-255 范围
                image_array = (image_array * 255).astype(np.uint8)
                # 创建 PIL Image
                pil_image = Image.fromarray(image_array)
            else:
                pil_image = image
            
            # 确保图像是 RGB 模式
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # 并行生成多张图片
            all_images = []
            import concurrent.futures
            import time
            
            def generate_single_i2i(image_index):
                try:
                    # 为每个线程创建独立的 VisualService 实例
                    thread_visual_service = create_visual_service_with_timeout(timeout)
                    
                    # 设置 API 密钥
                    if api_key and secret_key:
                        thread_visual_service.set_ak(api_key)
                        thread_visual_service.set_sk(secret_key)
                    else:
                        # 尝试从环境变量获取
                        env_ak = os.getenv('VOLC_AK')
                        env_sk = os.getenv('VOLC_SK')
                        if env_ak and env_sk:
                            thread_visual_service.set_ak(env_ak)
                            thread_visual_service.set_sk(env_sk)
                        else:
                            raise Exception("火山引擎 API Key 或 Secret Key 未设置")
                    
                    # 为每张图片生成不同的种子
                    current_seed = seed if seed != -1 else int(time.time() * 1000) + image_index
                    if current_seed > 99999999:
                        current_seed = current_seed % 99999999
                    
                    # 将图像转换为 base64
                    buffer = io.BytesIO()
                    pil_image.save(buffer, format='PNG')
                    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    
                    # 构建请求参数
                    form = {
                        "req_key": req_key,
                        "prompt": prompt,
                        "seed": current_seed,
                        "scale": scale,
                        "binary_data_base64": [image_base64]
                    }
                    
                    # 调用火山引擎 API
                    resp = thread_visual_service.cv_process(form)
                    
                    # 检查响应状态
                    if not resp or 'data' not in resp:
                        raise Exception(f"API 响应无效: {resp}")
                    
                    resp_data = resp['data']
                    
                    # 检查是否有错误
                    if 'error' in resp_data:
                        raise Exception(f"API 错误: {resp_data['error']}")
                    
                    # 获取图像数据（优先检查 image_urls，然后是 binary_data_base64）
                    if 'image_urls' in resp_data and resp_data['image_urls']:
                        # 如果有 URL，下载图像
                        image_url = resp_data['image_urls'][0]
                        response = requests.get(image_url)
                        if response.status_code == 200:
                            image_data = response.content
                        else:
                            raise Exception(f"下载图像失败: {response.status_code}")
                    elif 'binary_data_base64' in resp_data and resp_data['binary_data_base64']:
                        # 如果有 base64 数据，直接解码
                        binary_data_base64 = resp_data['binary_data_base64'][0]
                        image_data = base64.b64decode(binary_data_base64)
                    else:
                        raise Exception("响应中未找到图像数据")
                    
                    # 创建图像
                    result_image = Image.open(io.BytesIO(image_data))
                    
                    # 转换为 RGB 模式（ComfyUI 需要）
                    if result_image.mode != 'RGB':
                        result_image = result_image.convert('RGB')
                    
                    # 转换为 ComfyUI 图像格式
                    result_array = np.array(result_image).astype(np.float32) / 255.0
                    
                    return result_array
                    
                except Exception as e:
                    print(f"生成第 {image_index + 1} 张图片失败: {str(e)}")
                    # 返回错误图像
                    if isinstance(image, torch.Tensor):
                        if len(image.shape) == 4:
                            height, width = image.shape[2], image.shape[3]
                        else:
                            height, width = image.shape[1], image.shape[2]
                    else:
                        height, width = 512, 512
                    
                    error_image = np.zeros((height, width, 3), dtype=np.float32)
                    error_image[:, :, 0] = 1.0  # 红色背景表示错误
                    return error_image
            
            # 使用线程池并行执行
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(num_images, 4)) as executor:
                # 提交所有任务
                future_to_index = {executor.submit(generate_single_i2i, i): i for i in range(num_images)}
                
                # 收集结果
                for future in concurrent.futures.as_completed(future_to_index):
                    image_array = future.result()
                    all_images.append(image_array)
            
            # 将所有图片合并为一个批次
            if all_images:
                # 确保所有图片都有相同的形状
                first_image = all_images[0]
                for i, img in enumerate(all_images):
                    if img.shape != first_image.shape:
                        # 如果形状不匹配，调整到第一张图片的形状
                        temp_img = Image.fromarray((img * 255).astype(np.uint8))
                        temp_img = temp_img.resize((first_image.shape[1], first_image.shape[0]))
                        all_images[i] = np.array(temp_img).astype(np.float32) / 255.0
                
                # 堆叠所有图片
                batch_images = np.stack(all_images, axis=0)
                
                # 转换为 PyTorch 张量（ComfyUI 需要）
                result_tensor = torch.from_numpy(batch_images)
                
                # 构建信息字符串
                info = f"图生图成功！共 {len(all_images)} 张图片，缩放: {scale}, 种子: {seed}, 模型: {req_key}"
                
                return (result_tensor, info)
            else:
                raise Exception("没有成功生成任何图片")
            
        except Exception as e:
            error_msg = f"图生图失败: {str(e)}"
            print(error_msg)
            
            raise Exception(error_msg)


class MaletteVolcanoSmartAPI:
    """火山引擎智能图像生成节点 - 自动选择 T2I 或 I2I"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "一只可爱的小猫", "multiline": True}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 4096, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 4096, "step": 64}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
            },
            "optional": {
                "image_url": ("STRING", {"default": "", "placeholder": "图片URL（留空则使用T2I，有URL则使用I2I）"}),
                "scale": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.1}),
                "t2i_req_key": ("STRING", {"default": "jimeng_t2i_v31", "options": ["jimeng_t2i_v31", "jimeng_t2i_v30"]}),
                "i2i_req_key": ("STRING", {"default": "jimeng_i2i_v30", "options": ["jimeng_i2i_v30"]}),
                "api_key": ("STRING", {"default": "", "placeholder": "火山引擎 API Key"}),
                "secret_key": ("STRING", {"default": "", "placeholder": "火山引擎 Secret Key"}),
                "timeout": ("INT", {"default": 120, "min": 30, "max": 600, "step": 10}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "smart_generate"
    CATEGORY = "image generation"
    
    def smart_generate(self, prompt, width, height, seed, num_images, image_url="", scale=0.5, 
                      t2i_req_key="jimeng_t2i_v31", i2i_req_key="jimeng_i2i_v30", 
                      api_key="", secret_key="", timeout=120):
        try:
            # 检查是否有图片URL
            if image_url and image_url.strip():
                # 有URL，使用I2I模式
                return self._generate_i2i_from_url(prompt, width, height, seed, num_images, 
                                                 image_url, scale, i2i_req_key, api_key, secret_key, timeout)
            else:
                # 没有URL，使用T2I模式
                return self._generate_t2i(prompt, width, height, seed, num_images, 
                                        t2i_req_key, api_key, secret_key, timeout)
                
        except Exception as e:
            error_msg = f"智能生成失败: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)
    
    def _generate_t2i(self, prompt, width, height, seed, num_images, req_key, api_key, secret_key, timeout=120):
        """T2I 生成方法"""
        try:
            # 初始化火山引擎服务
            visual_service = create_visual_service_with_timeout(timeout)
            
            # 设置 API 密钥
            if api_key and secret_key:
                visual_service.set_ak(api_key)
                visual_service.set_sk(secret_key)
            else:
                env_ak = os.getenv('VOLC_AK')
                env_sk = os.getenv('VOLC_SK')
                if env_ak and env_sk:
                    visual_service.set_ak(env_ak)
                    visual_service.set_sk(env_sk)
                else:
                    raise Exception("火山引擎 API Key 或 Secret Key 未设置")

            # 并行生成多张图片
            all_images = []
            import concurrent.futures
            import time
            
            def generate_single_t2i(image_index):
                try:
                    # 为每个线程创建独立的 VisualService 实例
                    thread_visual_service = create_visual_service_with_timeout(timeout)
                    
                    # 设置 API 密钥
                    if api_key and secret_key:
                        thread_visual_service.set_ak(api_key)
                        thread_visual_service.set_sk(secret_key)
                    else:
                        env_ak = os.getenv('VOLC_AK')
                        env_sk = os.getenv('VOLC_SK')
                        if env_ak and env_sk:
                            thread_visual_service.set_ak(env_ak)
                            thread_visual_service.set_sk(env_sk)
                        else:
                            raise Exception("火山引擎 API Key 或 Secret Key 未设置")
                    
                    # 为每张图片生成不同的种子
                    current_seed = seed if seed != -1 else int(time.time() * 1000) + image_index
                    if current_seed > 99999999:
                        current_seed = current_seed % 99999999
                    
                    # 构建请求参数
                    form = {
                        "req_key": req_key,
                        "prompt": prompt,
                        "seed": current_seed,
                        "width": width,
                        "height": height
                    }
                    
                    # 调用火山引擎 API
                    resp = thread_visual_service.cv_process(form)
                    
                    # 检查响应状态
                    if not resp or 'data' not in resp:
                        raise Exception(f"API 响应无效: {resp}")
                    
                    resp_data = resp['data']
                    
                    # 检查是否有错误
                    if 'error' in resp_data:
                        raise Exception(f"API 错误: {resp_data['error']}")
                    
                    # 获取图像数据
                    if 'binary_data_base64' not in resp_data or not resp_data['binary_data_base64']:
                        raise Exception("响应中未找到图像数据")
                    
                    binary_data_base64 = resp_data['binary_data_base64'][0]

                    # 解码并创建图像
                    image_data = base64.b64decode(binary_data_base64)
                    pil_image = Image.open(io.BytesIO(image_data))
                    
                    # 转换为 RGB 模式
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                    
                    # 转换为 ComfyUI 图像格式
                    image_array = np.array(pil_image).astype(np.float32) / 255.0
                    
                    return image_array
                    
                except Exception as e:
                    print(f"生成第 {image_index + 1} 张图片失败: {str(e)}")
                    # 返回错误图像
                    error_image = np.zeros((height, width, 3), dtype=np.float32)
                    error_image[:, :, 0] = 1.0  # 红色背景表示错误
                    return error_image
            
            # 使用线程池并行执行
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(num_images, 4)) as executor:
                future_to_index = {executor.submit(generate_single_t2i, i): i for i in range(num_images)}
                
                for future in concurrent.futures.as_completed(future_to_index):
                    image_array = future.result()
                    all_images.append(image_array)
            
            # 将所有图片合并为一个批次
            if all_images:
                # 确保所有图片都有相同的形状
                first_image = all_images[0]
                for i, img in enumerate(all_images):
                    if img.shape != first_image.shape:
                        temp_img = Image.fromarray((img * 255).astype(np.uint8))
                        temp_img = temp_img.resize((first_image.shape[1], first_image.shape[0]))
                        all_images[i] = np.array(temp_img).astype(np.float32) / 255.0
                
                # 堆叠所有图片
                batch_images = np.stack(all_images, axis=0)
                
                # 转换为 PyTorch 张量
                image_tensor = torch.from_numpy(batch_images)
                
                # 构建信息字符串
                info = f"T2I 生成成功！共 {len(all_images)} 张图片，尺寸: {width}x{height}, 种子: {seed}, 模型: {req_key}"
                
                return (image_tensor, info)
            else:
                raise Exception("没有成功生成任何图片")
                
        except Exception as e:
            raise Exception(f"T2I 生成失败: {str(e)}")
    
    def _generate_i2i_from_url(self, prompt, width, height, seed, num_images, image_url, scale, req_key, api_key, secret_key, timeout=120):
        """从URL进行I2I生成"""
        try:
            # 下载图片一次，转换为base64
            print(f"正在下载图片: {image_url}")
            response = requests.get(image_url, timeout=timeout)
            if response.status_code != 200:
                raise Exception(f"下载图片失败: HTTP {response.status_code}")
            
            # 创建PIL图像
            pil_image = Image.open(io.BytesIO(response.content))
            
            # 确保图像是 RGB 模式
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # 调整图像尺寸以匹配目标尺寸
            if pil_image.size != (width, height):
                pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)
                print(f"调整图像尺寸: {pil_image.size}")
            
            img_width, img_height = pil_image.size
            print(f"图像尺寸: {img_width}x{img_height}")
            # 将图像转换为 base64（只转换一次）
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            print("图片下载完成，开始生成...")
            
            # 初始化火山引擎服务
            visual_service = create_visual_service_with_timeout(timeout)
            
            # 设置 API 密钥
            if api_key and secret_key:
                visual_service.set_ak(api_key)
                visual_service.set_sk(secret_key)
            else:
                env_ak = os.getenv('VOLC_AK')
                env_sk = os.getenv('VOLC_SK')
                if env_ak and env_sk:
                    visual_service.set_ak(env_ak)
                    visual_service.set_sk(env_sk)
                else:
                    raise Exception("火山引擎 API Key 或 Secret Key 未设置")
            
            # 并行生成多张图片
            all_images = []
            import concurrent.futures
            import time
            
            def generate_single_i2i_from_url(image_index):
                try:
                    # 为每个线程创建独立的 VisualService 实例
                    thread_visual_service = create_visual_service_with_timeout(timeout)
                    
                    # 设置 API 密钥
                    if api_key and secret_key:
                        thread_visual_service.set_ak(api_key)
                        thread_visual_service.set_sk(secret_key)
                    else:
                        env_ak = os.getenv('VOLC_AK')
                        env_sk = os.getenv('VOLC_SK')
                        if env_ak and env_sk:
                            thread_visual_service.set_ak(env_ak)
                            thread_visual_service.set_sk(env_sk)
                        else:
                            raise Exception("火山引擎 API Key 或 Secret Key 未设置")
                    
                    # 为每张图片生成不同的种子
                    current_seed = seed if seed != -1 else int(time.time() * 1000) + image_index
                    if current_seed > 99999999:
                        current_seed = current_seed % 99999999
                    # 构建请求参数 - 使用已下载的base64数据
                    form = {
                        "req_key": req_key,
                        "prompt": prompt,
                        "width": img_width,
                        "height": img_height,
                        "seed": current_seed,
                        "scale": scale,
                        "binary_data_base64": [image_base64]  # 使用已转换的base64，不重复下载
                    }
                    
                    # 调用火山引擎 API
                    resp = thread_visual_service.cv_process(form)
                    
                    # 检查响应状态
                    if not resp or 'data' not in resp:
                        raise Exception(f"API 响应无效: {resp}")
                    
                    resp_data = resp['data']
                    
                    # 检查是否有错误
                    if 'error' in resp_data:
                        raise Exception(f"API 错误: {resp_data['error']}")
                    
                    # 获取图像数据
                    if 'image_urls' in resp_data and resp_data['image_urls']:
                        # 如果有 URL，下载图像
                        result_image_url = resp_data['image_urls'][0]
                        result_response = requests.get(result_image_url, timeout=timeout)
                        if result_response.status_code == 200:
                            image_data = result_response.content
                        else:
                            raise Exception(f"下载结果图像失败: {result_response.status_code}")
                    elif 'binary_data_base64' in resp_data and resp_data['binary_data_base64']:
                        # 如果有 base64 数据，直接解码
                        binary_data_base64 = resp_data['binary_data_base64'][0]
                        image_data = base64.b64decode(binary_data_base64)
                    else:
                        raise Exception("响应中未找到图像数据")
                    
                    # 创建图像
                    result_image = Image.open(io.BytesIO(image_data))
                    
                    # 转换为 RGB 模式
                    if result_image.mode != 'RGB':
                        result_image = result_image.convert('RGB')
                    
                    # 转换为 ComfyUI 图像格式
                    result_array = np.array(result_image).astype(np.float32) / 255.0
                    
                    return result_array
                    
                except Exception as e:
                    print(f"生成第 {image_index + 1} 张图片失败: {str(e)}")
                    # 返回错误图像
                    error_image = np.zeros((height, width, 3), dtype=np.float32)
                    error_image[:, :, 0] = 1.0  # 红色背景表示错误
                    return error_image
            
            # 使用线程池并行执行
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(num_images, 4)) as executor:
                future_to_index = {executor.submit(generate_single_i2i_from_url, i): i for i in range(num_images)}
                
                for future in concurrent.futures.as_completed(future_to_index):
                    image_array = future.result()
                    all_images.append(image_array)
            
            # 将所有图片合并为一个批次
            if all_images:
                # 确保所有图片都有相同的形状
                first_image = all_images[0]
                for i, img in enumerate(all_images):
                    if img.shape != first_image.shape:
                        temp_img = Image.fromarray((img * 255).astype(np.uint8))
                        temp_img = temp_img.resize((first_image.shape[1], first_image.shape[0]))
                        all_images[i] = np.array(temp_img).astype(np.float32) / 255.0
                
                # 堆叠所有图片
                batch_images = np.stack(all_images, axis=0)
                
                # 转换为 PyTorch 张量
                result_tensor = torch.from_numpy(batch_images)
                
                # 构建信息字符串
                info = f"I2I 生成成功！共 {len(all_images)} 张图片，尺寸: {width}x{height}, 缩放: {scale}, 种子: {seed}, 模型: {req_key}"
                
                return (result_tensor, info)
            else:
                raise Exception("没有成功生成任何图片")
                
        except Exception as e:
            raise Exception(f"I2I 生成失败: {str(e)}")


# 节点注册
NODE_CLASS_MAPPINGS = {
    "MaletteVolcanoT2IAPI": MaletteVolcanoT2IAPI,
    "MaletteVolcanoI2IAPI": MaletteVolcanoI2IAPI,
    "MaletteVolcanoSmartAPI": MaletteVolcanoSmartAPI
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaletteVolcanoT2IAPI": "火山引擎 T2I API",
    "MaletteVolcanoI2IAPI": "火山引擎 I2I API",
    "MaletteVolcanoSmartAPI": "火山引擎 智能生成 API"
}