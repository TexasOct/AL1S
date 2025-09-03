"""
OpenAI服务模块
"""
import asyncio
from typing import List, Optional, Dict, Any
from openai import AsyncOpenAI
from loguru import logger

from ..config import config
from ..models import Message, ChatResponse, Role


class OpenAIService:
    """OpenAI服务类"""
    
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=config.openai.api_key,
            base_url=config.openai.base_url,
            timeout=config.openai.timeout
        )
        self.model = config.openai.model
        self.max_tokens = config.openai.max_tokens
        self.temperature = config.openai.temperature
    
    async def chat_completion(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """调用OpenAI聊天完成API"""
        try:
            # 构建完整的消息列表
            full_messages = messages.copy()
            
            # 调用OpenAI API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            if response.choices and response.choices[0].message:
                return response.choices[0].message.content
            else:
                logger.error("OpenAI API返回空响应")
                return None
                
        except Exception as e:
            logger.error(f"OpenAI API调用失败: {e}")
            return None
    
    async def generate_role_prompt(self, description: str) -> str:
        """生成角色提示词"""
        try:
            # 构建系统提示词
            system_prompt = f"你是一个智能、友好的AI助手，能够帮助用户解决各种问题。"
            
            # 构建用户消息
            user_message = f"请根据以下描述生成一个角色设定：{description}"
            
            # 调用OpenAI API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            if response.choices and response.choices[0].message:
                return response.choices[0].message.content
            else:
                return f"你是一个{description}。"
                
        except Exception as e:
            logger.error(f"生成角色提示词失败: {e}")
            return f"你是一个{description}。"
    
    async def analyze_image(self, image_url: str, prompt: str = "请描述这张图片") -> str:
        """分析图片内容"""
        try:
            # 检查当前模型是否支持图片分析
            if "vision" in self.model.lower() or "gpt-4" in self.model.lower():
                # 使用支持图片的模型（OpenAI格式）
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": image_url}}
                            ]
                        }
                    ],
                    max_tokens=500,
                    temperature=0.7
                )
            elif "kimi" in self.model.lower() or "moonshot" in self.model.lower():
                # 使用 Moonshot 的 files API 进行图片分析
                try:
                    # 下载图片数据
                    import aiohttp
                    async with aiohttp.ClientSession() as session:
                        async with session.get(image_url) as resp:
                            if resp.status == 200:
                                image_data = await resp.read()
                                
                                # 创建文件对象
                                from pathlib import Path
                                import tempfile
                                import os
                                
                                # 创建临时文件
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                                    temp_file.write(image_data)
                                    temp_file_path = temp_file.name
                                
                                try:
                                    # 使用 files API 上传并分析图片
                                    file_object = await self.client.files.create(
                                        file=Path(temp_file_path), 
                                        purpose="file-extract"
                                    )
                                    
                                    # 获取文件内容
                                    file_content = await self.client.files.content(file_id=file_object.id)
                                    
                                    # 构建消息进行图片分析
                                    messages = [
                                        {
                                            "role": "system",
                                            "content": "你是一个专业的图片分析助手，请根据图片内容回答用户的问题。"
                                        },
                                        {
                                            "role": "system",
                                            "content": file_content
                                        },
                                        {"role": "user", "content": prompt}
                                    ]
                                    
                                    # 调用 chat completion
                                    response = await self.client.chat.completions.create(
                                        model=self.model,
                                        messages=messages,
                                        max_tokens=500,
                                        temperature=0.7
                                    )
                                    
                                    if response.choices and response.choices[0].message:
                                        return response.choices[0].message.content
                                    else:
                                        return "无法分析图片内容"
                                        
                                finally:
                                    # 清理临时文件
                                    if os.path.exists(temp_file_path):
                                        os.unlink(temp_file_path)
                                        
                            else:
                                return f"无法下载图片，HTTP状态码: {resp.status}"
                                
                except Exception as e:
                    logger.error(f"Moonshot files API 图片分析失败: {e}")
                    return f"图片分析失败: {str(e)}"
            else:
                # 对于不支持图片的模型，返回提示信息
                logger.warning(f"当前模型 {self.model} 不支持图片分析")
                return f"当前使用的模型 {self.model} 不支持图片分析功能。请使用支持图片的模型。"
                
        except Exception as e:
            logger.error(f"图片分析失败: {e}")
            return "图片分析失败，请稍后再试"
    