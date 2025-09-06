"""
LangChain Agent 服务
- 基于 LangChain 的完整 Agent 实现
- 集成向量存储服务进行知识检索
- 集成学习服务进行自主学习
- 支持工具调用和复杂推理
- 提供与统一 Agent 服务相同的接口
"""

import asyncio
import json
from typing import Any, Dict, List, Optional

from loguru import logger

from ..config import config
from ..infra.vector import VectorService
from ..services.learning_service import LearningService


class LangChainAgentService:
    """基于 LangChain 的完整 Agent 服务"""

    def __init__(
        self,
        database_service=None,
        mcp_service=None,
        vector_store_path: str = "data/vector_store",
    ):
        self.database_service = database_service
        self.mcp_service = mcp_service

        # 核心服务组件
        self.vector_service = VectorService(database_service, vector_store_path)
        self.learning_service = None  # 将在初始化时创建

        # LangChain 组件
        self._llm = None
        self._agent = None
        self._tools = []
        self._retriever_tool = None

        # 状态
        self._initialized = False
        self._current_conversation_id = None

    async def initialize(self) -> bool:
        """初始化 LangChain Agent 服务"""
        try:
            # 初始化向量服务
            logger.info("正在初始化向量服务...")

            # 确定嵌入模型类型
            embedding_model_type = "tfidf"  # 默认值
            if hasattr(config, "agent") and hasattr(config.agent, "embedding_model"):
                embedding_model_type = config.agent.embedding_model
            elif hasattr(config, "langchain") and hasattr(
                config.langchain, "embedding_model_name"
            ):
                embedding_model_type = config.langchain.embedding_model_name

            # 确定向量存储后端
            vector_store_backend = "memory"  # 默认值
            if hasattr(config, "agent") and hasattr(config.agent, "vector_store"):
                vector_store_backend = config.agent.vector_store
            elif hasattr(config, "langchain") and hasattr(
                config.langchain, "vector_store"
            ):
                vector_store_backend = config.langchain.vector_store

            vector_success = await self.vector_service.initialize(
                embedding_model_type=embedding_model_type,
                vector_store_backend=vector_store_backend,
            )

            if not vector_success:
                logger.error("向量服务初始化失败")
                return False

            # 初始化 LLM
            logger.info("正在初始化 LangChain LLM...")
            from langchain_openai import ChatOpenAI

            self._llm = ChatOpenAI(
                model=config.openai.model,
                temperature=config.openai.temperature,
                openai_api_key=config.openai.api_key,
                base_url=config.openai.base_url,
                timeout=config.openai.timeout,
            )

            # 初始化学习服务
            self.learning_service = LearningService(
                self.database_service,
                self.vector_service,
                self._create_llm_adapter(),  # 创建 LLM 适配器
            )

            # 初始化工具
            await self._initialize_tools()

            # 构建 Agent
            await self._build_agent()

            self._initialized = True
            logger.info("LangChain Agent 服务初始化完成")
            return True

        except Exception as e:
            logger.error(f"LangChain Agent 服务初始化失败: {e}")
            return False

    def _create_llm_adapter(self):
        """创建 LLM 适配器，用于学习服务"""

        class LLMAdapter:
            def __init__(self, langchain_llm):
                self.langchain_llm = langchain_llm

            async def chat_completion(self, messages, **kwargs):
                """适配学习服务的 chat_completion 接口"""
                try:
                    # 将消息转换为 LangChain 格式
                    from langchain_core.messages import HumanMessage, SystemMessage

                    langchain_messages = []
                    for msg in messages:
                        if msg["role"] == "system":
                            langchain_messages.append(
                                SystemMessage(content=msg["content"])
                            )
                        elif msg["role"] == "user":
                            langchain_messages.append(
                                HumanMessage(content=msg["content"])
                            )

                    # 调用 LangChain LLM
                    response = await self.langchain_llm.ainvoke(langchain_messages)
                    return response.content

                except Exception as e:
                    logger.error(f"LLM 适配器调用失败: {e}")
                    return None

        return LLMAdapter(self._llm)

    async def _initialize_tools(self):
        """初始化 Agent 工具"""
        try:
            from langchain_core.tools import Tool

            # 创建知识检索工具
            self._retriever_tool = Tool(
                name="knowledge_search",
                description="搜索用户相关的个人信息、历史对话记录、生日、喜好等私人数据。当用户询问关于自己的信息、生日、个人设置或历史记录时，必须使用此工具。输入应该是搜索关键词。",
                func=self._search_knowledge_sync,
            )

            self._tools = [self._retriever_tool]

            # 添加 MCP 工具（如果可用）
            if self.mcp_service:
                mcp_tools = await self._create_mcp_tools()
                self._tools.extend(mcp_tools)

            logger.info(f"初始化了 {len(self._tools)} 个工具")

        except Exception as e:
            logger.error(f"初始化工具失败: {e}")

    async def _create_mcp_tools(self) -> List:
        """创建 MCP 工具"""
        try:
            from langchain_core.tools import Tool

            tools = []
            available_tools = self.mcp_service.get_available_tools()

            if not available_tools:
                return tools

            for tool_name, tool_info in available_tools.items():
                # 创建 LangChain 工具包装器
                def make_mcp_tool(name: str):
                    def mcp_tool_func(query: str) -> str:
                        try:
                            # 尝试解析参数
                            try:
                                args = json.loads(query)
                            except json.JSONDecodeError:
                                args = {"query": query}

                            # 在同步上下文中运行异步 MCP 调用
                            import asyncio

                            try:
                                loop = asyncio.get_running_loop()
                                # 如果在事件循环中，使用 run_coroutine_threadsafe
                                future = asyncio.run_coroutine_threadsafe(
                                    self.mcp_service.call_tool(name, args), loop
                                )
                                result = future.result(timeout=30)
                            except RuntimeError:
                                # 没有运行中的事件循环，直接运行
                                result = asyncio.run(
                                    self.mcp_service.call_tool(name, args)
                                )

                            return result or f"工具 {name} 执行完成"
                        except Exception as e:
                            return f"工具 {name} 执行失败: {str(e)}"

                    return mcp_tool_func

                tool = Tool(
                    name=tool_name,
                    description=tool_info.get("description", f"MCP工具: {tool_name}"),
                    func=make_mcp_tool(tool_name),
                )
                tools.append(tool)

            logger.info(f"创建了 {len(tools)} 个 MCP 工具")
            return tools

        except Exception as e:
            logger.error(f"创建 MCP 工具失败: {e}")
            return []

    def _search_knowledge_sync(self, query: str) -> str:
        """同步知识搜索（供 LangChain 工具使用）"""
        try:
            import asyncio
            import concurrent.futures

            def run_async_in_thread():
                """在新线程中运行异步函数"""
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(
                        self._search_knowledge_async(query)
                    )
                finally:
                    new_loop.close()

            # 在线程池中运行异步函数
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_async_in_thread)
                return future.result(timeout=30)

        except Exception as e:
            logger.error(f"同步知识搜索失败: {e}")
            return f"知识搜索失败: {str(e)}"

    async def _search_knowledge_async(self, query: str) -> str:
        """异步知识搜索"""
        try:
            results = await self.vector_service.search_knowledge(query, top_k=5)

            if not results:
                return "没有找到相关知识。"

            # 格式化搜索结果
            formatted_results = []
            for i, result in enumerate(results[:3], 1):  # 最多返回3个结果
                title = result.get("title", "无标题")
                content = result.get("content", result.get("summary", ""))
                score = result.get("similarity_score", 0)

                formatted_results.append(
                    f"{i}. {title}\n内容: {content[:200]}...\n相似度: {score:.2f}"
                )

            return "\n\n".join(formatted_results)

        except Exception as e:
            logger.error(f"异步知识搜索失败: {e}")
            return f"知识搜索失败: {str(e)}"

    async def _build_agent(self):
        """构建 LangChain Agent"""
        try:
            from langchain.agents import AgentExecutor, create_openai_tools_agent
            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

            # 创建 Agent 提示模板
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """你是天童爱丽丝，一个智能助手，具有搜索用户个人信息的能力。

🔍 CRITICAL: 工具使用规则
1. 当用户询问关于自己的任何信息时，必须首先使用 knowledge_search 工具搜索
2. 包括但不限于：生日、个人喜好、历史对话、个人设置、之前提到的信息
3. 搜索关键词：从用户问题中提取关键词，如"生日"、"喜好"、"信息"等
4. 基于搜索结果回答，如果没找到则说明没有相关记录

⚡ 触发搜索的问题类型：
- "我的生日是什么时候？" → 搜索"生日"
- "你知道我的信息吗？" → 搜索"用户信息"  
- "我之前说过什么？" → 搜索"历史对话"
- "我喜欢什么？" → 搜索"喜好"

记住：你是天童爱丽丝，活泼可爱，用"邦邦卡邦"等口头禅。""",
                    ),
                    ("user", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ]
            )

            # 创建 Agent
            agent = create_openai_tools_agent(self._llm, self._tools, prompt)

            # 创建 Agent 执行器
            self._agent = AgentExecutor(
                agent=agent,
                tools=self._tools,
                verbose=True,
                max_iterations=5,
                max_execution_time=60,
                handle_parsing_errors=True,
            )

            logger.info("LangChain Agent 构建完成")

        except Exception as e:
            logger.error(f"构建 Agent 失败: {e}")
            # 创建简化的回退方案
            self._agent = None

    def set_conversation_id(self, conversation_id: int):
        """设置当前对话ID，用于工具调用记录"""
        self._current_conversation_id = conversation_id

    async def chat_completion(
        self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]] = None
    ) -> Optional[str]:
        """聊天完成接口（与统一 Agent 服务兼容）"""
        try:
            if not self._initialized:
                logger.warning("LangChain Agent 服务未初始化")
                return None

            # 提取系统消息（包含角色信息）和用户消息
            system_message = ""
            user_message = ""
            conversation_history = []

            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                elif msg["role"] == "user":
                    user_message = msg["content"]
                elif msg["role"] == "assistant":
                    # 保存对话历史用于上下文
                    conversation_history.append(msg)

            if not user_message:
                return "抱歉，无法从消息中提取用户问题。"

            # 如果有 Agent，使用 Agent 处理
            if self._agent:
                try:
                    # 构建包含角色信息的输入
                    agent_input = {"input": user_message}

                    # 如果有系统消息（角色信息），添加到聊天历史中
                    if system_message:
                        # 将角色信息作为额外上下文传递给 Agent
                        enhanced_input = (
                            f"角色设定: {system_message}\n\n用户问题: {user_message}"
                        )
                        agent_input = {"input": enhanced_input}

                    response = await self._agent.ainvoke(agent_input)
                    return response.get("output", "抱歉，Agent 未能生成有效回答。")
                except Exception as e:
                    logger.warning(f"Agent 处理失败，使用简化模式: {e}")

            # 简化模式：直接使用 LLM + 知识检索
            return await self._simple_rag_response(
                user_message, messages, system_message
            )

        except Exception as e:
            logger.error(f"LangChain Agent 聊天完成失败: {e}")
            return None

    async def _simple_rag_response(
        self,
        user_message: str,
        messages: List[Dict[str, str]],
        system_message: str = "",
    ) -> str:
        """简化的 RAG 响应模式"""
        try:
            # 搜索相关知识
            knowledge_results = await self.vector_service.search_knowledge(
                user_message, top_k=3
            )

            # 构建上下文
            context = ""
            if knowledge_results:
                context = "\n相关知识:\n"
                for i, result in enumerate(knowledge_results, 1):
                    title = result.get("title", "无标题")
                    content = result.get("content", result.get("summary", ""))
                    context += f"{i}. {title}: {content[:200]}...\n"

            # 构建增强的消息
            enhanced_messages = []

            # 添加系统消息（角色信息）
            if system_message:
                enhanced_messages.append({"role": "system", "content": system_message})

            # 添加对话历史（除了最后一条用户消息）
            for msg in messages[:-1]:
                if msg["role"] != "system":  # 避免重复添加系统消息
                    enhanced_messages.append(msg)

            # 增强最后一条用户消息
            enhanced_user_message = user_message
            if context:
                enhanced_user_message = f"{user_message}\n\n{context}"

            enhanced_messages.append({"role": "user", "content": enhanced_user_message})

            # 调用 LLM
            from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

            langchain_messages = []
            for msg in enhanced_messages:
                if msg["role"] == "system":
                    langchain_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    langchain_messages.append(AIMessage(content=msg["content"]))

            response = await self._llm.ainvoke(langchain_messages)
            return response.content

        except Exception as e:
            logger.error(f"简化 RAG 响应失败: {e}")
            return "抱歉，处理您的请求时出现了问题。"

    async def analyze_image(
        self, image_data: bytes, prompt: str = "请描述这张图片"
    ) -> Optional[str]:
        """图片分析功能"""
        try:
            # 使用 OpenAI 的视觉模型
            import base64

            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                api_key=config.openai.api_key, base_url=config.openai.base_url
            )

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

            # 调用 OpenAI API
            response = await client.chat.completions.create(
                model="gpt-4-vision-preview",
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
        self,
        user_message: str,
        bot_response: str,
        conversation_id: int,
        user_id: int,
        conversation_context: List[Dict[str, str]] = None,
    ) -> int:
        """从对话中学习（使用学习服务）"""
        try:
            if not self.learning_service:
                logger.warning("学习服务未初始化")
                return 0

            return await self.learning_service.learn_from_conversation(
                user_message,
                bot_response,
                conversation_id,
                user_id,
                conversation_context,
            )

        except Exception as e:
            logger.error(f"从对话中学习失败: {e}")
            return 0

    async def get_learning_statistics(self) -> Dict[str, Any]:
        """获取学习统计信息"""
        try:
            if not self.learning_service:
                return {}

            return await self.learning_service.get_learning_statistics()

        except Exception as e:
            logger.error(f"获取学习统计失败: {e}")
            return {}

    async def optimize_knowledge_base(self) -> Dict[str, int]:
        """优化知识库"""
        try:
            if not self.learning_service:
                return {}

            return await self.learning_service.optimize_knowledge_base()

        except Exception as e:
            logger.error(f"优化知识库失败: {e}")
            return {}

    async def learn_from_conversation(
        self,
        user_message: str,
        bot_response: str,
        conversation_id: int = None,
        user_id: int = None,
    ):
        """从对话中学习（与统一 Agent 服务兼容）"""
        try:
            if self.learning_service:
                await self.learning_service.learn_from_conversation(
                    user_message, bot_response, conversation_id, user_id
                )
                logger.info("LangChain Agent 完成对话学习")
            else:
                logger.debug("学习服务未初始化，跳过学习")
        except Exception as e:
            logger.error(f"LangChain Agent 学习失败: {e}")

    def cleanup(self):
        """清理资源"""
        try:
            if self.vector_service:
                self.vector_service.cleanup()

            if self.learning_service:
                self.learning_service.cleanup()

            if self._llm:
                del self._llm
                self._llm = None

            if self._agent:
                del self._agent
                self._agent = None

            self._tools.clear()

            logger.debug("LangChain Agent 服务资源清理完成")

        except Exception as e:
            logger.debug(f"LangChain Agent 服务资源清理失败: {e}")

    def __del__(self):
        """析构函数"""
        try:
            self.cleanup()
        except Exception:
            pass
