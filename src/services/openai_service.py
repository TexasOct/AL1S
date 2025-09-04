"""
OpenAI服务模块
"""
import asyncio
import json
from typing import List, Optional, Dict, Any, Callable
from openai import AsyncOpenAI
from loguru import logger

from ..config import config
from ..models import Message, ChatResponse, Role


class OpenAIService:
    """OpenAI服务类"""
    
    def __init__(self, tool_handler: Optional[Callable] = None, database_service=None):
        self.client = AsyncOpenAI(
            api_key=config.openai.api_key,
            base_url=config.openai.base_url,
            timeout=config.openai.timeout
        )
        self.model = config.openai.model
        self.max_tokens = config.openai.max_tokens
        self.temperature = config.openai.temperature
        self.tool_handler = tool_handler  # MCP工具处理器
        self.database_service = database_service  # 数据库服务
        self._current_conversation_id = None  # 当前对话ID，用于工具调用记录
    
    def set_conversation_id(self, conversation_id: int):
        """设置当前对话ID，用于工具调用记录"""
        self._current_conversation_id = conversation_id
    
    async def chat_completion(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]] = None) -> Optional[str]:
        """调用OpenAI聊天完成API，支持工具调用"""
        try:
            # 验证和清理消息列表
            full_messages = []
            for msg in messages:
                # 确保消息有必要的字段且内容不为空
                if (isinstance(msg, dict) and 
                    "role" in msg and 
                    "content" in msg and 
                    msg["content"] and 
                    str(msg["content"]).strip()):
                    
                    cleaned_msg = {
                        "role": msg["role"],
                        "content": str(msg["content"]).strip()
                    }
                    full_messages.append(cleaned_msg)
                else:
                    logger.warning(f"跳过无效消息: {msg}")
            
            # 确保至少有一条消息
            if not full_messages:
                logger.error("没有有效的消息可以发送到OpenAI API")
                return "抱歉，消息处理出现问题，请重新发送。"
            
            # 构建API调用参数
            api_params = {
                "model": self.model,
                "messages": full_messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }
            
            # 如果有工具，添加工具参数
            if tools and self.tool_handler:
                api_params["tools"] = tools
                api_params["tool_choice"] = "auto"
            
            # 调用OpenAI API
            response = await self.client.chat.completions.create(**api_params)
            
            if response.choices and response.choices[0].message:
                message = response.choices[0].message
                
                # 检查是否有工具调用
                if hasattr(message, 'tool_calls') and message.tool_calls and self.tool_handler:
                    return await self._handle_tool_calls(message, full_messages, tools)
                else:
                    return message.content
            else:
                logger.error("OpenAI API返回空响应")
                return None
                
        except Exception as e:
            logger.error(f"OpenAI API调用失败: {e}")
            return None
    
    async def _handle_tool_calls(self, message, original_messages: List[Dict[str, str]], tools: List[Dict[str, Any]]) -> Optional[str]:
        """处理工具调用"""
        try:
            # 添加助手的工具调用消息
            tool_call_message = {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]
            }
            original_messages.append(tool_call_message)
            
            # 执行每个工具调用
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                logger.info(f"执行工具调用: {function_name} 参数: {function_args}")
                
                # 记录工具调用开始时间
                import time
                start_time = time.time()
                success = True
                error_message = None
                
                try:
                    # 调用MCP工具
                    tool_result = await self.tool_handler(function_name, function_args)
                except Exception as e:
                    tool_result = f"工具调用失败: {str(e)}"
                    success = False
                    error_message = str(e)
                    logger.error(f"工具调用失败: {e}")
                
                # 计算执行时间
                execution_time = time.time() - start_time
                
                # 记录工具调用到数据库
                if self.database_service and hasattr(self, '_current_conversation_id'):
                    try:
                        await self.database_service.record_tool_call(
                            conversation_id=self._current_conversation_id,
                            tool_name=function_name,
                            arguments=function_args,
                            result=str(tool_result) if tool_result else None,
                            success=success,
                            error_message=error_message,
                            execution_time=execution_time
                        )
                    except Exception as e:
                        logger.warning(f"记录工具调用失败: {e}")
                
                # 添加工具结果消息
                tool_result_message = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(tool_result) if tool_result else "工具执行完成"
                }
                original_messages.append(tool_result_message)
            
            # 再次调用LLM以获取最终响应
            final_response = await self.client.chat.completions.create(
                model=self.model,
                messages=original_messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                tools=tools,
                tool_choice="auto"
            )
            
            if final_response.choices and final_response.choices[0].message:
                final_message = final_response.choices[0].message
                
                # 检查是否有更多工具调用（递归处理）
                if hasattr(final_message, 'tool_calls') and final_message.tool_calls:
                    return await self._handle_tool_calls(final_message, original_messages, tools)
                else:
                    return final_message.content
            else:
                logger.error("工具调用后的最终响应为空")
                return "工具执行完成，但无法生成响应"
                
        except Exception as e:
            logger.error(f"处理工具调用失败: {e}")
            return f"工具调用处理失败: {str(e)}"
    
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
    