"""
火山引擎方舟平台视频生成 API 节点
支持 Seedance 1.5 pro、1.0 pro、1.0 lite 等模型
所有视频生成都是异步的，需要轮询获取结果
"""

import base64
import io
import os
import requests
import json
import time
from PIL import Image
import numpy as np
import torch
import subprocess
from pathlib import Path
from io import BytesIO

# 尝试导入 VideoFromFile，如果不可用则使用字符串 URL
try:
    from comfy_api.input_impl import VideoFromFile
    VIDEO_FROM_FILE_AVAILABLE = True
except ImportError:
    VIDEO_FROM_FILE_AVAILABLE = False
    print("[Volcano] VideoFromFile 不可用，将使用 URL 字符串返回视频")


class ArkVideoGenerationNode:
    """方舟平台视频生成节点 - 完整功能"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ([
                    "doubao-seedance-1-5-pro-251215",
                    "doubao-seedance-1-0-pro-250528", 
                    "doubao-seedance-1-0-pro-fast-251015",
                    "doubao-seedance-1-0-lite-i2v-250428",
                    "doubao-seedance-1-0-lite-t2v-250428"
                ],),
                "prompt": ("STRING", {"default": "一只小猫在草地上玩耍", "multiline": True}),
                "duration": ("INT", {"default": 5, "min": 2, "max": 12}),
                "resolution": (["480p", "720p", "1080p"],),
                "ratio": (["16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "adaptive"],),
            },
            "optional": {
                "image_first": ("IMAGE",),
                "image_last": ("IMAGE",),
                "image_url_first": ("STRING", {"default": "", "placeholder": "首帧图片URL"}),
                "image_url_last": ("STRING", {"default": "", "placeholder": "尾帧图片URL"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 4294967295}),
                "fps": ("INT", {"default": 24, "min": 24, "max": 24}),
                "watermark": ("BOOLEAN", {"default": False}),
                "camera_fixed": ("BOOLEAN", {"default": False}),
                "return_last_frame": ("BOOLEAN", {"default": False}),
                "service_tier": (["default", "flex"],),
                "generate_audio": ("BOOLEAN", {"default": False}),
                "api_key": ("STRING", {"default": "", "placeholder": "方舟平台 API Key"}),
                "timeout": ("INT", {"default": 300, "min": 60, "max": 1800, "step": 60}),
                "poll_interval": ("INT", {"default": 5, "min": 2, "max": 30, "step": 1}),
                "download_video": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING" if not VIDEO_FROM_FILE_AVAILABLE else "VIDEO")
    RETURN_NAMES = ("image", "info", "video_url", "video")
    FUNCTION = "generate_video"
    CATEGORY = "Volcano/Ark"
    
    def generate_video(self, model, prompt, duration, resolution, ratio, image_first=None, image_last=None,
                      image_url_first="", image_url_last="", seed=-1, fps=24, watermark=False,
                      camera_fixed=False, return_last_frame=False, service_tier="default",
                      generate_audio=False, api_key="", timeout=300, poll_interval=5, download_video=False):
        try:
            # 获取 API Key
            if not api_key or not api_key.strip():
                api_key = os.getenv('ARK_API_KEY')
                if not api_key:
                    raise Exception("未设置方舟平台 API Key，请在节点参数或环境变量 ARK_API_KEY 中设置")
            
            # API 端点
            api_url = "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks"
            
            # 构建请求头
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # 构建文本提示词（包含参数命令）
            text_prompt = prompt
            
            # 构建参数命令
            params = []
            params.append(f"--rs {resolution}")
            params.append(f"--rt {ratio}")
            if duration > 0:
                params.append(f"--dur {duration}")
            params.append(f"--fps {fps}")
            if watermark:
                params.append("--wm true")
            if camera_fixed:
                params.append("--cf true")
            if seed != -1:
                params.append(f"--seed {seed}")
            
            text_prompt += " " + " ".join(params)
            
            # 构建请求体
            content = [
                {
                    "type": "text",
                    "text": text_prompt
                }
            ]
            
            # 处理图片输入
            has_first_frame = False
            has_last_frame = False
            
            # 处理首帧图片
            if image_first is not None:
                first_frame_base64 = self._image_to_base64(image_first)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{first_frame_base64}"
                    },
                    "role": "first_frame"
                })
                has_first_frame = True
            elif image_url_first and image_url_first.strip():
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_url_first.strip()
                    },
                    "role": "first_frame"
                })
                has_first_frame = True
            
            # 处理尾帧图片（首尾帧生成模式）
            if image_last is not None:
                last_frame_base64 = self._image_to_base64(image_last)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{last_frame_base64}"
                    },
                    "role": "last_frame"
                })
                has_last_frame = True
            elif image_url_last and image_url_last.strip():
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_url_last.strip()
                    },
                    "role": "last_frame"
                })
                has_last_frame = True
            
            # 构建完整请求体
            request_body = {
                "model": model,
                "content": content,
                "service_tier": service_tier,
                "return_last_frame": return_last_frame
            }
            
            # Seedance 1.5 pro 支持音频控制
            if "1-5" in model:
                request_body["generate_audio"] = generate_audio
            
            # 确定生成模式
            if has_first_frame and has_last_frame:
                mode_desc = "首尾帧生成"
            elif has_first_frame:
                mode_desc = "首帧生成"
            else:
                mode_desc = "文生视频"
            
            print(f"开始调用方舟平台视频生成 API")
            print(f"模型: {model}, 模式: {mode_desc}")
            print(f"分辨率: {resolution}, 宽高比: {ratio}, 时长: {duration}s")
            
            # 发送请求
            response = requests.post(api_url, headers=headers, json=request_body, timeout=timeout)
            
            if response.status_code != 200:
                error_msg = f"提交视频生成任务失败: {response.status_code} - {response.text}"
                raise Exception(error_msg)
            
            result = response.json()
            
            if 'error' in result:
                error_info = result['error']
                raise Exception(f"API 错误 [{error_info.get('code', 'unknown')}]: {error_info.get('message', 'unknown error')}")
            
            if 'id' not in result:
                raise Exception("响应中未找到任务 ID")
            
            task_id = result['id']
            print(f"任务已提交，task_id: {task_id}")
            
            # 轮询获取结果
            video_url, video_info = self._poll_video_result(task_id, api_key, timeout, poll_interval)
            
            print(f"视频生成成功！")
            print(f"视频 URL: {video_url}")
            
            # 下载视频（如果需要）
            video_object = None
            if download_video and video_url:
                video_object = self._download_video(video_url, timeout)
            
            # 下载视频第一帧作为输出
            if video_url:
                # 使用 ffmpeg 提取第一帧
                frame_image = self._extract_frame_from_video(video_url, timeout)
                if frame_image is not None:
                    info = f"[{mode_desc}] 视频生成完成 - {video_info}"
                    video_output = video_object if video_object else video_url
                    return (frame_image, info, video_url, video_output)
            
            # 如果无法提取帧，返回黑色占位图
            placeholder = np.zeros((720, 1280, 3), dtype=np.float32)
            placeholder[:, :, 0] = 0.2  # 灰色背景
            placeholder = np.expand_dims(placeholder, 0)
            placeholder_tensor = torch.from_numpy(placeholder)
            
            info = f"[{mode_desc}] 视频生成完成（无法提取预览） - {video_info}"
            video_output = video_object if video_object else video_url
            return (placeholder_tensor, info, video_url, video_output)
            
        except Exception as e:
            error_msg = f"视频生成失败: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)
    
    def _image_to_base64(self, image):
        """将 ComfyUI 图像转换为 base64"""
        if isinstance(image, torch.Tensor):
            if len(image.shape) == 4:
                image = image[0]
            image_array = image.cpu().numpy()
            image_array = (image_array * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_array)
        else:
            pil_image = image
        
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def _poll_video_result(self, task_id, api_key, timeout, poll_interval):
        """轮询视频生成结果"""
        api_url = f"https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks/{task_id}"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        start_time = time.time()
        
        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout:
                raise Exception(f"轮询超时: 超过 {timeout} 秒")
            
            try:
                response = requests.get(api_url, headers=headers, timeout=timeout)
                
                if response.status_code != 200:
                    print(f"查询失败: HTTP {response.status_code}，继续轮询...")
                    time.sleep(poll_interval)
                    continue
                
                result = response.json()
                
                if 'error' in result:
                    print(f"查询出错: {result['error']}，继续轮询...")
                    time.sleep(poll_interval)
                    continue
                
                # 检查任务状态
                status = result.get('status', '')
                print(f"任务状态: {status}")
                
                if status == 'succeeded':
                    # 任务成功
                    content = result.get('content', {})
                    video_url = content.get('video_url', '')
                    duration = result.get('duration', 'unknown')
                    ratio = result.get('ratio', 'unknown')
                    resolution = result.get('resolution', 'unknown')
                    
                    video_info = f"分辨率: {resolution}, 宽高比: {ratio}, 时长: {duration}s"
                    
                    return video_url, video_info
                    
                elif status == 'failed' or status == 'expired':
                    error_msg = result.get('error', '任务执行失败')
                    raise Exception(f"任务执行失败: {error_msg} - {status}")
                
                elif status in ['queued', 'running']:
                    # 任务还在处理中
                    print(f"任务处理中（{status}），等待 {poll_interval} 秒后重试...")
                    time.sleep(poll_interval)
                    continue
                
                else:
                    # 未知状态
                    print(f"未知状态: {status}，继续轮询...")
                    time.sleep(poll_interval)
                    continue
                    
            except Exception as e:
                elapsed_time = time.time() - start_time
                if elapsed_time > timeout:
                    raise Exception(f"轮询超时: 超过 {timeout} 秒，最后一次请求出错: {str(e)}")
                
                print(f"轮询请求失败: {str(e)}，继续重试...")
                time.sleep(poll_interval)
                continue
    
    def _extract_frame_from_video(self, video_url, timeout):
        """使用 ffmpeg 从视频URL提取第一帧"""
        try:
            # 检查是否安装了 ffmpeg
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
            if result.returncode != 0:
                print("警告: ffmpeg 未安装，无法提取视频帧")
                return None
            
            # 下载视频文件到临时目录
            temp_dir = Path("/tmp") if os.path.exists("/tmp") else Path(".")
            video_file = temp_dir / "temp_video.mp4"
            frame_file = temp_dir / "temp_frame.png"
            
            print(f"正在下载视频...")
            response = requests.get(video_url, timeout=timeout)
            if response.status_code != 200:
                print(f"下载视频失败: HTTP {response.status_code}")
                return None
            
            with open(video_file, 'wb') as f:
                f.write(response.content)
            
            # 提取第一帧
            print(f"正在提取视频第一帧...")
            subprocess.run([
                'ffmpeg', '-i', str(video_file), 
                '-vf', 'select=eq(n\\,0)',
                '-q:v', '2',
                str(frame_file)
            ], capture_output=True, timeout=30)
            
            if frame_file.exists():
                # 读取帧并转换为 ComfyUI 格式
                pil_image = Image.open(frame_file)
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                
                image_array = np.array(pil_image).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_array)
                image_tensor = image_tensor.unsqueeze(0)  # 添加批次维度
                
                # 清理临时文件
                video_file.unlink(missing_ok=True)
                frame_file.unlink(missing_ok=True)
                
                return image_tensor
            
        except Exception as e:
            print(f"提取视频帧失败: {str(e)}")
        
        return None
    
    def _download_video(self, video_url, timeout=300):
        """下载视频文件到 BytesIO"""
        try:
            print(f"[Volcano] 开始下载视频: {video_url}")
            response = requests.get(video_url, timeout=timeout, stream=True)
            response.raise_for_status()
            
            video_data = BytesIO()
            for chunk in response.iter_content(chunk_size=8192):
                video_data.write(chunk)
            
            video_data.seek(0)
            print(f"[Volcano] 视频下载完成，大小: {len(video_data.getvalue())} 字节")
            
            # 如果 VideoFromFile 可用，返回 VideoFromFile 对象
            if VIDEO_FROM_FILE_AVAILABLE:
                return VideoFromFile(video_data)
            else:
                # 否则返回 BytesIO 对象（但实际上会返回 URL）
                print("[Volcano] VideoFromFile 不可用，返回 BytesIO")
                return video_data
            
        except Exception as e:
            error_msg = f"下载视频失败: {str(e)}"
            print(f"[Volcano] {error_msg}")
            return None


class ArkVideoGenerationSmartNode:
    """方舟平台智能视频生成节点 - 自动判断生成模式"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "一只小猫在草地上玩耍", "multiline": True}),
            },
            "optional": {
                "image_first": ("IMAGE",),
                "image_last": ("IMAGE",),
                "image_url_first": ("STRING", {"default": "", "placeholder": "首帧图片URL"}),
                "image_url_last": ("STRING", {"default": "", "placeholder": "尾帧图片URL"}),
                "generate_audio": ("BOOLEAN", {"default": False}),
                "duration": ("INT", {"default": 5, "min": 2, "max": 12}),
                "resolution": (["480p", "720p", "1080p"],),
                "ratio": (["16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "adaptive"],),
                "seed": ("INT", {"default": -1, "min": -1, "max": 4294967295}),
                "watermark": ("BOOLEAN", {"default": False}),
                "return_last_frame": ("BOOLEAN", {"default": False}),
                "service_tier": (["default", "flex"],),
                "api_key": ("STRING", {"default": "", "placeholder": "方舟平台 API Key"}),
                "timeout": ("INT", {"default": 300, "min": 60, "max": 1800, "step": 60}),
                "poll_interval": ("INT", {"default": 5, "min": 2, "max": 30, "step": 1}),
                "download_video": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING" if not VIDEO_FROM_FILE_AVAILABLE else "VIDEO")
    RETURN_NAMES = ("image", "info", "video_url", "video")
    FUNCTION = "smart_generate"
    CATEGORY = "Volcano/Ark"
    
    def smart_generate(self, prompt, image_first=None, image_last=None, image_url_first="", 
                      image_url_last="", generate_audio=False, duration=5, resolution="720p", ratio="16:9",
                      seed=-1, watermark=False, return_last_frame=False, service_tier="default",
                      api_key="", timeout=300, poll_interval=5, download_video=False):
        try:
            # 智能判断生成模式
            has_first_frame = (image_first is not None) or (image_url_first and image_url_first.strip())
            has_last_frame = (image_last is not None) or (image_url_last and image_url_last.strip())
            
            # 自动选择最新的模型（Seedance 1.5 pro）
            model = "doubao-seedance-1-5-pro-251215"
            
            # 确定生成模式并输出提示
            if has_first_frame and has_last_frame:
                mode_desc = "首尾帧生成"
                print(f"[智能模式] 检测到首帧和尾帧输入，使用 {mode_desc}")
            elif has_first_frame:
                mode_desc = "首帧生成"
                print(f"[智能模式] 检测到首帧输入，使用 {mode_desc}")
            else:
                mode_desc = "文生视频"
                print(f"[智能模式] 未检测到图片输入，使用 {mode_desc}")
            
            # 调用完整节点
            ark_node = ArkVideoGenerationNode()
            result = ark_node.generate_video(
                model=model,
                prompt=prompt,
                duration=duration,
                resolution=resolution,
                ratio=ratio,
                image_first=image_first,
                image_last=image_last,
                image_url_first=image_url_first,
                image_url_last=image_url_last,
                seed=seed,
                fps=24,
                watermark=watermark,
                camera_fixed=False,
                return_last_frame=return_last_frame,
                service_tier=service_tier,
                generate_audio=generate_audio,
                api_key=api_key,
                timeout=timeout,
                poll_interval=poll_interval,
                download_video=download_video
            )
            
            # 在信息中添加模式标识
            image, info, video_url, video = result
            info = f"[智能-{mode_desc}] {info}"
            
            return (image, info, video_url, video)
            
        except Exception as e:
            error_msg = f"智能视频生成失败: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "ArkVideoGenerationNode": ArkVideoGenerationNode,
    "ArkVideoGenerationSmartNode": ArkVideoGenerationSmartNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkVideoGenerationNode": "方舟视频生成 (Ark Video Generation)",
    "ArkVideoGenerationSmartNode": "方舟智能视频生成 (Ark Smart Video)",
}
