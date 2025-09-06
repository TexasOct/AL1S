"""
统一 Agent 服务
- 整合 OpenAI、RAG 和 LangChain 功能的统一服务
- OpenAI API 调用 (替代 OpenAIService)
- RAG 知识检索 (替代 RAGService)
- LangChain Agent 能力
- MCP 工具集成
- 图片分析
"""

import asyncio
import json
from typing import Any, Dict, List, Optional

from loguru import logger

# OpenAI imports
from openai import AsyncOpenAI

from ..config import config
from ..infra.vector import VectorService
from ..services.learning_service import LearningService


class UnifiedAgentService:
    """统一 Agent 服务 - 整合 OpenAI、RAG 和 LangChain 功能"""

    def __init__(
        self,
        database_service=None,
        mcp_service=None,
        vector_store_path: str = "data/vector_store",
    ):
        # Core services
        self.database_service = database_service
        self.mcp_service = mcp_service

        # OpenAI client
        self.openai_client = AsyncOpenAI(
            api_key=config.openai.api_key,
            base_url=config.openai.base_url,
            timeout=config.openai.timeout,
        )

        # Vector service for RAG
        self.vector_service = VectorService(
            database_service=database_service, vector_store_path=vector_store_path
        )

        # Learning service
        self.learning_service = LearningService(
            llm_service=self,  # 使用自己作为 LLM 服务
            database_service=database_service,
            vector_service=self.vector_service,
        )

        # State
        self._initialized = False
        self._current_conversation_id = None

        # Tool handler for MCP integration
        self.tool_handler = self._handle_mcp_tool if mcp_service else None

    async def initialize(self) -> bool:
        """初始化统一服务"""
        try:
            # 初始化 RAG 组件
            await self._initialize_rag()

            self._initialized = True
            logger.info("统一 Agent 服务初始化完成")
            return True
        except Exception as e:
            logger.error(f"统一 Agent 服务初始化失败: {e}")
            return False

    async def _initialize_rag(self):
        """初始化 RAG 组件"""
        try:
            # 使用 agent 配置中的嵌入模型
            embedding_model = (
                config.agent.embedding_model
                if hasattr(config, "agent") and config.agent.embedding_model
                else config.rag.embedding_model
            )
            vector_store_backend = (
                config.agent.vector_store_backend
                if hasattr(config, "agent") and config.agent.vector_store_backend
                else config.rag.vector_store_backend
            )

            # 初始化 vector service
            await self.vector_service.initialize(
                embedding_model_type=embedding_model,
                vector_store_backend=vector_store_backend,
            )

            logger.info("RAG 组件初始化完成")
        except Exception as e:
            logger.error(f"RAG 组件初始化失败: {e}")
            raise

    def set_conversation_id(self, conversation_id: int):
        """设置当前对话ID，用于工具调用记录"""
        self._current_conversation_id = conversation_id

    async def chat_completion(
        self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]] = None
    ) -> Optional[str]:
        """统一的聊天完成接口"""
        try:
            # 验证和清理消息列表
            full_messages = []
            for msg in messages:
                if (
                    isinstance(msg, dict)
                    and "role" in msg
                    and "content" in msg
                    and msg["content"]
                    and str(msg["content"]).strip()
                ):

                    cleaned_msg = {
                        "role": msg["role"],
                        "content": str(msg["content"]).strip(),
                    }
                    full_messages.append(cleaned_msg)
                else:
                    logger.warning(f"跳过无效消息: {msg}")

            # 确保至少有一条消息
            if not full_messages:
                logger.error("没有有效的消息可以发送到OpenAI API")
                return "抱歉，消息处理出现问题，请重新发送。"

            # 如果启用了 RAG，增强用户查询
            user_query = None
            for msg in reversed(full_messages):
                if msg["role"] == "user":
                    user_query = msg["content"]
                    break

            if user_query and self._rag_index is not None:
                rag_context = await self._retrieve_knowledge(user_query)
                if rag_context:
                    # 增强最后一条用户消息
                    enhanced_query = f"{user_query}\n\n相关知识:\n{rag_context}"
                    for msg in reversed(full_messages):
                        if msg["role"] == "user":
                            msg["content"] = enhanced_query
                            break

            # 构建API调用参数
            api_params = {
                "model": config.openai.model,
                "messages": full_messages,
                "max_tokens": config.openai.max_tokens,
                "temperature": config.openai.temperature,
            }

            # 如果有工具，添加工具参数
            if tools and self.tool_handler:
                api_params["tools"] = tools
                api_params["tool_choice"] = "auto"

            # 调用OpenAI API
            response = await self.openai_client.chat.completions.create(**api_params)

            if response.choices and response.choices[0].message:
                message = response.choices[0].message

                # 处理工具调用
                if message.tool_calls and self.tool_handler:
                    return await self._handle_tool_calls(
                        message.tool_calls, full_messages
                    )

                return message.content

            return None

        except Exception as e:
            logger.error(f"聊天完成失败: {e}")
            return None

    async def _retrieve_knowledge(self, query: str, top_k: int = 3) -> str:
        """检索相关知识"""
        try:
            # 使用 vector_service 进行知识搜索
            results = await self.vector_service.search_knowledge(query, top_k=top_k)

            if not results:
                return ""

            # 构建上下文
            context_parts = []
            for result in results:
                title = result.get("title", "")
                summary = result.get("summary", result.get("content", ""))
                if title or summary:
                    context_parts.append(f"{title}: {summary}")

            return "\n".join(context_parts)

        except Exception as e:
            logger.error(f"知识检索失败: {e}")
            return ""

    async def _handle_tool_calls(
        self, tool_calls, messages: List[Dict[str, str]]
    ) -> str:
        """处理工具调用"""
        try:
            tool_results = []

            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}

                # 调用MCP工具
                result = await self.tool_handler(tool_name, arguments)

                tool_results.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": tool_name,
                        "content": result or "工具执行完成",
                    }
                )

            # 添加工具结果到消息历史
            messages.extend(tool_results)

            # 再次调用OpenAI API获取最终回复
            response = await self.openai_client.chat.completions.create(
                model=config.openai.model,
                messages=messages,
                max_tokens=config.openai.max_tokens,
                temperature=config.openai.temperature,
            )

            if response.choices and response.choices[0].message:
                return response.choices[0].message.content

            return "工具调用完成，但没有生成回复。"

        except Exception as e:
            logger.error(f"处理工具调用失败: {e}")
            return f"工具调用过程中出现错误: {str(e)}"

    async def _handle_mcp_tool(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Optional[str]:
        """MCP工具处理器"""
        if not self.mcp_service:
            return None

        try:
            result = await self.mcp_service.call_tool(tool_name, arguments)

            # 记录工具调用到数据库
            if self.database_service and self._current_conversation_id:
                await asyncio.to_thread(
                    self.database_service.save_tool_call,
                    self._current_conversation_id,
                    tool_name,
                    arguments,
                    result,
                )

            return result

        except Exception as e:
            logger.error(f"MCP工具调用失败: {e}")
            return f"工具调用失败: {str(e)}"

    async def analyze_image(
        self, image_data: bytes, prompt: str = "请描述这张图片"
    ) -> Optional[str]:
        """图片分析"""
        try:
            import base64

            # 编码图片
            image_base64 = base64.b64encode(image_data).decode("utf-8")

            # 构建消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
                }
            ]

            # 调用OpenAI API
            response = await self.openai_client.chat.completions.create(
                model="gpt-4-vision-preview",  # 使用支持图片的模型
                messages=messages,
                max_tokens=config.openai.max_tokens,
            )

            if response.choices and response.choices[0].message:
                return response.choices[0].message.content

            return None

        except Exception as e:
            logger.error(f"图片分析失败: {e}")
            return None

    async def learn_from_conversation(
        self, user_message: str, bot_response: str, conversation_id: int, user_id: int
    ):
        """从对话中学习（自动知识提取）"""
        try:
            # 使用 learning_service 的学习功能
            await self.learning_service.learn_from_conversation(
                user_message=user_message,
                bot_response=bot_response,
                conversation_id=conversation_id,
                user_id=user_id,
            )

        except Exception as e:
            logger.error(f"从对话中学习失败: {e}")

    def cleanup(self):
        """清理资源"""
        try:
            # 清理 vector_service
            if hasattr(self.vector_service, "cleanup"):
                self.vector_service.cleanup()

            logger.debug("统一 Agent 服务资源清理完成")
        except Exception as e:
            logger.debug(f"统一 Agent 服务资源清理失败: {e}")

    def __del__(self):
        """析构函数"""
        try:
            self.cleanup()
        except Exception:
            pass
