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
import tempfile
from pathlib import Path


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
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "fps": ("INT", {"default": 24, "min": 24, "max": 24}),
                "watermark": ("BOOLEAN", {"default": False}),
                "camera_fixed": ("BOOLEAN", {"default": False}),
                "return_last_frame": ("BOOLEAN", {"default": False}),
                "service_tier": (["default", "flex"],),
                "generate_audio": ("BOOLEAN", {"default": False}),
                "api_key": ("STRING", {"default": "", "placeholder": "方舟平台 API Key"}),
                "timeout": ("INT", {"default": 300, "min": 60, "max": 1800, "step": 60}),
                "poll_interval": ("INT", {"default": 5, "min": 2, "max": 30, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "info", "video_url")
    FUNCTION = "generate_video"
    CATEGORY = "Volcano/Ark"
    
    def generate_video(self, model, prompt, duration, resolution, ratio, image_first=None, image_last=None,
                      image_url_first="", image_url_last="", seed=-1, fps=24, watermark=False,
                      camera_fixed=False, return_last_frame=False, service_tier="default",
                      generate_audio=False, api_key="", timeout=300, poll_interval=5):
        try:
            if seed > 4294967295:
                seed = seed % 4294967296
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
            
            # 提取视频第一帧作为预览
            frame_image = None
            if video_url:
                frame_image = self._extract_frame_from_video(video_url, timeout)
            
            if frame_image is None:
                placeholder = np.zeros((720, 1280, 3), dtype=np.float32)
                placeholder[:, :, 0] = 0.2
                frame_image = torch.from_numpy(np.expand_dims(placeholder, 0))
                info = f"[{mode_desc}] 视频生成完成（无法提取预览） - {video_info}"
            else:
                info = f"[{mode_desc}] 视频生成完成 - {video_info}"
            
            return (frame_image, info, video_url)
            
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
            # 检查 ffmpeg 是否可用
            result = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
            if result.returncode != 0:
                print("[Volcano] ffmpeg 未安装，跳过帧提取")
                return None
            
            # 使用 tempfile 确保跨平台兼容
            temp_dir = Path(tempfile.gettempdir())
            timestamp = int(time.time())
            video_file = temp_dir / f"volcano_video_{timestamp}.mp4"
            frame_file = temp_dir / f"volcano_frame_{timestamp}.png"
            
            try:
                print("[Volcano] 下载视频用于帧提取...")
                response = requests.get(video_url, timeout=timeout)
                if response.status_code != 200:
                    print(f"[Volcano] 下载视频失败: HTTP {response.status_code}")
                    return None
                
                with open(video_file, "wb") as f:
                    f.write(response.content)
                
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(video_file),
                     "-vframes", "1", "-q:v", "2", str(frame_file)],
                    capture_output=True, timeout=30,
                )
                
                if frame_file.exists():
                    pil_image = Image.open(frame_file)
                    if pil_image.mode != "RGB":
                        pil_image = pil_image.convert("RGB")
                    image_array = np.array(pil_image).astype(np.float32) / 255.0
                    return torch.from_numpy(image_array).unsqueeze(0)
            finally:
                if video_file.exists():
                    video_file.unlink(missing_ok=True)
                if frame_file.exists():
                    frame_file.unlink(missing_ok=True)
            
        except Exception as e:
            print(f"[Volcano] 提取视频帧失败: {str(e)}")
        
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
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "watermark": ("BOOLEAN", {"default": False}),
                "return_last_frame": ("BOOLEAN", {"default": False}),
                "service_tier": (["default", "flex"],),
                "api_key": ("STRING", {"default": "", "placeholder": "方舟平台 API Key"}),
                "timeout": ("INT", {"default": 300, "min": 60, "max": 1800, "step": 60}),
                "poll_interval": ("INT", {"default": 5, "min": 2, "max": 30, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "info", "video_url")
    FUNCTION = "smart_generate"
    CATEGORY = "Volcano/Ark"
    
    def smart_generate(self, prompt, image_first=None, image_last=None, image_url_first="", 
                      image_url_last="", generate_audio=False, duration=5, resolution="720p", ratio="16:9",
                      seed=-1, watermark=False, return_last_frame=False, service_tier="default",
                      api_key="", timeout=300, poll_interval=5):
        try:
            if seed > 4294967295:
                seed = seed % 4294967296
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
            )
            
            # 在信息中添加模式标识
            image, info, video_url = result
            info = f"[智能-{mode_desc}] {info}"
            
            return (image, info, video_url)
            
        except Exception as e:
            error_msg = f"智能视频生成失败: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)


class ArkVideoGenerationV2Node:
    """方舟平台通用视频生成节点 V2
    
    支持 Seedance 2.0 及所有旧版模型，核心能力：
    - 多图参考 (reference_image)：batch IMAGE 输入，每张图作为一个 reference_image
    - 视频参考 (reference_video)：通过 URL 输入参考视频
    - 音频参考 (reference_audio)：通过 URL 输入参考音频
    - 首帧/尾帧 (first_frame/last_frame)：兼容旧版 Seedance 1.0/1.5 模型
    - 文生视频：纯文本提示词生成
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("STRING", {"default": "doubao-seedance-2-0-260128",
                                     "placeholder": "模型名称，如 doubao-seedance-2-0-260128"}),
                "prompt": ("STRING", {"default": "一只小猫在草地上玩耍", "multiline": True}),
                "api_version": (["v2", "v1"],),
            },
            "optional": {
                # --- 参考图片（V2: reference_image，支持 batch 多图） ---
                "ref_images": ("IMAGE",),
                # --- 首帧图片（V1: first_frame / V2: reference_image） ---
                "first_frame": ("IMAGE",),
                # --- 尾帧图片（V1: last_frame） ---
                "last_frame": ("IMAGE",),
                # --- URL 输入（支持多个，换行分隔） ---
                "image_urls": ("STRING", {"default": "", "multiline": True,
                                          "placeholder": "参考图片URL，多个用换行分隔"}),
                "video_urls": ("STRING", {"default": "", "multiline": True,
                                          "placeholder": "参考视频URL，多个用换行分隔"}),
                "audio_urls": ("STRING", {"default": "", "multiline": True,
                                          "placeholder": "参考音频URL，多个用换行分隔"}),
                "first_frame_url": ("STRING", {"default": "", "placeholder": "首帧图片URL"}),
                "last_frame_url": ("STRING", {"default": "", "placeholder": "尾帧图片URL"}),
                # --- 生成参数 ---
                "duration": ("INT", {"default": 5, "min": 2, "max": 12}),
                "ratio": (["16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "adaptive"],),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "watermark": ("BOOLEAN", {"default": False}),
                "generate_audio": ("BOOLEAN", {"default": False}),
                # --- 旧模型专用参数 ---
                "resolution": (["720p", "480p", "1080p"],),
                "fps": ("INT", {"default": 24, "min": 24, "max": 24}),
                "camera_fixed": ("BOOLEAN", {"default": False}),
                "return_last_frame": ("BOOLEAN", {"default": False}),
                "service_tier": (["default", "flex"],),
                # --- 通用配置 ---
                "api_key": ("STRING", {"default": "", "placeholder": "方舟平台 API Key"}),
                "timeout": ("INT", {"default": 600, "min": 60, "max": 1800, "step": 60}),
                "poll_interval": ("INT", {"default": 10, "min": 2, "max": 60, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("preview_frame", "info", "video_url")
    FUNCTION = "generate"
    CATEGORY = "Volcano/Ark"

    def generate(self, model, prompt, api_version="v2",
                 ref_images=None, first_frame=None, last_frame=None,
                 image_urls="", video_urls="", audio_urls="",
                 first_frame_url="", last_frame_url="",
                 duration=5, ratio="16:9", seed=-1, watermark=False, generate_audio=False,
                 resolution="720p", fps=24, camera_fixed=False, return_last_frame=False,
                 service_tier="default", api_key="", timeout=600, poll_interval=10):
        try:
            if seed > 4294967295:
                seed = seed % 4294967296

            api_key = self._resolve_api_key(api_key)
            is_v2_model = (api_version == "v2")

            # 构建 content 列表
            content = self._build_content(
                prompt=prompt,
                is_v2_model=is_v2_model,
                ref_images=ref_images,
                first_frame=first_frame,
                last_frame=last_frame,
                image_urls=image_urls,
                video_urls=video_urls,
                audio_urls=audio_urls,
                first_frame_url=first_frame_url,
                last_frame_url=last_frame_url,
                resolution=resolution,
                ratio=ratio,
                duration=duration,
                fps=fps,
                watermark=watermark,
                camera_fixed=camera_fixed,
                seed=seed,
            )

            # 构建请求体
            request_body = {"model": model, "content": content}

            if is_v2_model:
                # Seedance 2.0：参数放在请求体顶层
                request_body["ratio"] = ratio
                request_body["duration"] = duration
                request_body["watermark"] = watermark
                request_body["generate_audio"] = generate_audio
            else:
                # 旧模型：额外字段
                request_body["service_tier"] = service_tier
                request_body["return_last_frame"] = return_last_frame
                if "1-5" in model:
                    request_body["generate_audio"] = generate_audio

            mode_desc = self._describe_mode(content)
            print(f"[ArkVideoV2] 模型: {model} | 模式: {mode_desc}")
            print(f"[ArkVideoV2] 比例: {ratio} | 时长: {duration}s | 音频: {generate_audio}")

            # 提交任务
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            api_url = "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks"

            response = requests.post(api_url, headers=headers, json=request_body, timeout=60)
            if response.status_code != 200:
                raise Exception(f"提交任务失败: HTTP {response.status_code} - {response.text}")

            result = response.json()
            if "error" in result:
                error_info = result["error"]
                raise Exception(f"API 错误 [{error_info.get('code', '?')}]: {error_info.get('message', '?')}")
            if "id" not in result:
                raise Exception(f"响应中未找到任务 ID: {json.dumps(result, ensure_ascii=False)[:500]}")

            task_id = result["id"]
            print(f"[ArkVideoV2] 任务已提交 task_id={task_id}")

            # 轮询结果
            video_url, video_info = self._poll_video_result(task_id, api_key, timeout, poll_interval)
            print(f"[ArkVideoV2] 视频生成成功: {video_url}")

            # 提取预览帧
            preview_frame = self._extract_frame_from_video(video_url, timeout)
            if preview_frame is None:
                preview_frame = self._create_placeholder_frame()

            info = f"[{mode_desc}] {video_info}"
            return (preview_frame, info, video_url)

        except Exception as e:
            error_msg = f"视频生成失败: {str(e)}"
            print(f"[ArkVideoV2] {error_msg}")
            raise Exception(error_msg)

    # ----------------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------------

    def _resolve_api_key(self, api_key):
        """获取 API Key，优先使用参数传入，其次环境变量"""
        if api_key and api_key.strip():
            return api_key.strip()
        env_key = os.getenv("ARK_API_KEY")
        if env_key:
            return env_key
        raise Exception("未设置方舟平台 API Key，请在节点参数或环境变量 ARK_API_KEY 中设置")

    def _build_content(self, prompt, is_v2_model,
                       ref_images, first_frame, last_frame,
                       image_urls, video_urls, audio_urls,
                       first_frame_url, last_frame_url,
                       resolution, ratio, duration, fps, watermark, camera_fixed, seed):
        """构建 API 请求的 content 列表"""
        content = []

        # --- 文本提示词 ---
        if is_v2_model:
            content.append({"type": "text", "text": prompt})
        else:
            text_prompt = prompt
            params = [f"--rs {resolution}", f"--rt {ratio}"]
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
            content.append({"type": "text", "text": text_prompt})

        # --- 参考图片（batch IMAGE → reference_image） ---
        if ref_images is not None and isinstance(ref_images, torch.Tensor):
            batch_size = ref_images.shape[0] if len(ref_images.shape) == 4 else 1
            for i in range(batch_size):
                single_image = ref_images[i] if len(ref_images.shape) == 4 else ref_images
                image_base64 = self._image_tensor_to_base64(single_image)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    "role": "reference_image",
                })

        # --- 参考图片 URL ---
        for url in self._parse_urls(image_urls):
            content.append({
                "type": "image_url",
                "image_url": {"url": url},
                "role": "reference_image",
            })

        # --- 首帧图片 ---
        if first_frame is not None and isinstance(first_frame, torch.Tensor):
            first_base64 = self._image_tensor_to_base64(
                first_frame[0] if len(first_frame.shape) == 4 else first_frame
            )
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{first_base64}"},
                "role": "first_frame",
            })
        elif first_frame_url and first_frame_url.strip():
            content.append({
                "type": "image_url",
                "image_url": {"url": first_frame_url.strip()},
                "role": "first_frame",
            })

        # --- 尾帧图片 ---
        if last_frame is not None and isinstance(last_frame, torch.Tensor):
            last_base64 = self._image_tensor_to_base64(
                last_frame[0] if len(last_frame.shape) == 4 else last_frame
            )
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{last_base64}"},
                "role": "last_frame",
            })
        elif last_frame_url and last_frame_url.strip():
            content.append({
                "type": "image_url",
                "image_url": {"url": last_frame_url.strip()},
                "role": "last_frame",
            })

        # --- 参考视频 URL ---
        for url in self._parse_urls(video_urls):
            content.append({
                "type": "video_url",
                "video_url": {"url": url},
                "role": "reference_video",
            })

        # --- 参考音频 URL ---
        for url in self._parse_urls(audio_urls):
            content.append({
                "type": "audio_url",
                "audio_url": {"url": url},
                "role": "reference_audio",
            })

        return content

    def _parse_urls(self, urls_text):
        """解析多行/逗号分隔的 URL 列表"""
        if not urls_text or not urls_text.strip():
            return []
        urls = []
        for line in urls_text.replace(",", "\n").split("\n"):
            url = line.strip()
            if url:
                urls.append(url)
        return urls

    def _image_tensor_to_base64(self, image_tensor):
        """将单张 ComfyUI 图像 tensor (H,W,C) 转换为 base64"""
        image_array = image_tensor.cpu().numpy()
        image_array = (image_array * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_array)
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _describe_mode(self, content):
        """根据 content 列表描述当前生成模式"""
        roles = [item.get("role", "") for item in content if item.get("type") != "text"]
        if not roles:
            return "文生视频"
        parts = []
        ref_image_count = roles.count("reference_image")
        ref_video_count = roles.count("reference_video")
        ref_audio_count = roles.count("reference_audio")
        has_first = "first_frame" in roles
        has_last = "last_frame" in roles
        if ref_image_count > 0:
            parts.append(f"{ref_image_count}张参考图")
        if ref_video_count > 0:
            parts.append(f"{ref_video_count}个参考视频")
        if ref_audio_count > 0:
            parts.append(f"{ref_audio_count}个参考音频")
        if has_first and has_last:
            parts.append("首尾帧")
        elif has_first:
            parts.append("首帧")
        elif has_last:
            parts.append("尾帧")
        return " + ".join(parts) if parts else "文生视频"

    def _poll_video_result(self, task_id, api_key, timeout, poll_interval):
        """轮询视频生成结果"""
        api_url = f"https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks/{task_id}"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise Exception(f"轮询超时: 已等待 {int(elapsed)}s，超过 {timeout}s 限制")

            try:
                response = requests.get(api_url, headers=headers, timeout=60)
                if response.status_code != 200:
                    print(f"[ArkVideoV2] 查询失败 HTTP {response.status_code}，{poll_interval}s 后重试...")
                    time.sleep(poll_interval)
                    continue

                result = response.json()
                if "error" in result:
                    error_info = result["error"]
                    # 区分致命错误和临时错误
                    error_code = error_info.get("code", "")
                    error_message = error_info.get("message", str(error_info))
                    if error_code in ("task_not_found", "invalid_task"):
                        raise Exception(f"任务查询错误 [{error_code}]: {error_message}")
                    print(f"[ArkVideoV2] 查询出错: {error_message}，继续轮询...")
                    time.sleep(poll_interval)
                    continue

                status = result.get("status", "")
                print(f"[ArkVideoV2] 任务状态: {status} ({int(elapsed)}s)")

                if status == "succeeded":
                    content = result.get("content", {})
                    video_url = content.get("video_url", "")
                    task_duration = result.get("duration", "?")
                    task_ratio = result.get("ratio", "?")
                    task_resolution = result.get("resolution", "?")
                    video_info = f"分辨率: {task_resolution}, 比例: {task_ratio}, 时长: {task_duration}s"
                    return video_url, video_info

                elif status in ("failed", "expired"):
                    error_msg = result.get("error", "任务执行失败")
                    raise Exception(f"任务 {status}: {error_msg}")

                elif status in ("queued", "running"):
                    time.sleep(poll_interval)
                    continue

                else:
                    print(f"[ArkVideoV2] 未知状态: {status}，继续轮询...")
                    time.sleep(poll_interval)
                    continue

            except requests.exceptions.RequestException as network_error:
                if time.time() - start_time > timeout:
                    raise Exception(f"轮询超时，最后网络错误: {str(network_error)}")
                print(f"[ArkVideoV2] 网络错误: {str(network_error)}，{poll_interval}s 后重试...")
                time.sleep(poll_interval)

    def _extract_frame_from_video(self, video_url, timeout):
        """使用 ffmpeg 从视频 URL 提取第一帧作为预览（跨平台）"""
        try:
            result = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
            if result.returncode != 0:
                print("[ArkVideoV2] ffmpeg 未安装，跳过帧提取")
                return None

            temp_dir = Path(tempfile.gettempdir())
            timestamp = int(time.time())
            video_file = temp_dir / f"ark_v2_video_{timestamp}.mp4"
            frame_file = temp_dir / f"ark_v2_frame_{timestamp}.png"

            try:
                print("[ArkVideoV2] 下载视频用于帧提取...")
                response = requests.get(video_url, timeout=timeout)
                if response.status_code != 200:
                    print(f"[ArkVideoV2] 下载视频失败: HTTP {response.status_code}")
                    return None

                with open(video_file, "wb") as f:
                    f.write(response.content)

                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(video_file),
                     "-vframes", "1", "-q:v", "2", str(frame_file)],
                    capture_output=True, timeout=30,
                )

                if frame_file.exists():
                    pil_image = Image.open(frame_file)
                    if pil_image.mode != "RGB":
                        pil_image = pil_image.convert("RGB")
                    image_array = np.array(pil_image).astype(np.float32) / 255.0
                    return torch.from_numpy(image_array).unsqueeze(0)
            finally:
                if video_file.exists():
                    video_file.unlink(missing_ok=True)
                if frame_file.exists():
                    frame_file.unlink(missing_ok=True)

        except Exception as e:
            print(f"[ArkVideoV2] 提取视频帧失败: {str(e)}")
        return None

    def _create_placeholder_frame(self):
        """创建灰色占位预览图"""
        placeholder = np.full((720, 1280, 3), 0.2, dtype=np.float32)
        return torch.from_numpy(placeholder).unsqueeze(0)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "ArkVideoGenerationNode": ArkVideoGenerationNode,
    "ArkVideoGenerationSmartNode": ArkVideoGenerationSmartNode,
    "ArkVideoGenerationV2Node": ArkVideoGenerationV2Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArkVideoGenerationNode": "方舟视频生成 (Ark Video Generation)",
    "ArkVideoGenerationSmartNode": "方舟智能视频生成 (Ark Smart Video)",
    "ArkVideoGenerationV2Node": "方舟视频生成 V2 (Ark Video V2)",
}
