"""
聊天处理器模块
"""
from typing import Optional
from telegram import Update
from telegram.ext import ContextTypes
from loguru import logger
import time
import re

from .base_handler import BaseHandler
from ..services.openai_service import OpenAIService
from ..services.conversation_service import ConversationService
from ..services.mcp_service import MCPService
from ..services.rag_service import RAGService
from ..services.knowledge_extractor import KnowledgeExtractor
from ..models import Message


class ChatHandler(BaseHandler):
    """聊天处理器"""
    
    def __init__(self, openai_service: OpenAIService, conversation_service: ConversationService, 
                 mcp_service: MCPService = None, database_service=None, rag_service: RAGService = None):
        super().__init__("ChatHandler", "处理用户聊天消息")
        self.openai_service = openai_service
        self.conversation_service = conversation_service
        self.mcp_service = mcp_service
        self.database_service = database_service
        self.rag_service = rag_service
        
        # 初始化知识提取器
        self.knowledge_extractor = KnowledgeExtractor(openai_service) if rag_service else None
    
    def can_handle(self, update: Update) -> bool:
        """检查是否可以处理此更新"""
        return (
            update.message is not None and
            update.message.text is not None and
            not update.message.text.startswith('/') and
            not update.message.text.startswith('!')
        )
    
    def _build_system_prompt(self, role) -> str:
        """构建系统提示词"""
        html_instructions = """
请使用以下HTML标签来格式化你的回复（仅使用这些标签）：
- <b>文本</b> - 粗体
- <i>文本</i> - 斜体  
- <u>文本</u> - 下划线
- <s>文本</s> - 删除线
- <code>代码</code> - 行内代码
- <pre>代码块</pre> - 代码块
- <a href="链接">文本</a> - 链接

不要使用其他HTML标签，不要使用Markdown语法。
"""
        
        if role and hasattr(role, 'personality'):
            return f"你是{role.name}。{role.personality}请自然地回复用户，不要提及你的角色设定或规则。\n\n{html_instructions}"
        return f"你是一个有用的AI助手，请自然地回复用户。\n\n{html_instructions}"
    
    def _build_system_prompt_with_rag(self, role, retrieved_knowledge) -> str:
        """构建包含RAG知识的系统提示词"""
        # 基础系统提示词
        base_prompt = self._build_system_prompt(role)
        
        # 如果没有检索到知识，返回基础提示词
        if not retrieved_knowledge:
            return base_prompt
        
        # 构建知识上下文
        knowledge_context = "\n\n=== 相关知识参考 ===\n"
        for i, (knowledge_entry, score) in enumerate(retrieved_knowledge[:3], 1):  # 只使用前3个最相关的
            knowledge_context += f"{i}. {knowledge_entry.title}\n"
            knowledge_context += f"内容: {knowledge_entry.content}\n"
            if knowledge_entry.keywords:
                knowledge_context += f"关键词: {knowledge_entry.keywords}\n"
            knowledge_context += f"相关性: {score:.2f}\n\n"
        
        knowledge_context += """请参考以上知识来回答用户的问题。如果相关知识能够帮助回答问题，请自然地融入到回复中。
如果相关知识与用户问题不太相关，可以忽略。不要直接提及"根据我的知识库"或类似表述，要让回复显得自然。
=== 知识参考结束 ===\n\n"""
        
        return base_prompt + knowledge_context
    
    def _get_placeholder_message(self, role) -> str:
        """根据角色生成个性化的占位信息"""
        if not role or not hasattr(role, 'name'):
            return "🤔 正在思考中..."
        
        # 根据角色名称生成个性化占位信息
        role_name = role.name.lower()
        
        if "爱丽丝" in role_name or "alice" in role_name:
            placeholders = [
                "🌸 爱丽丝正在思考呢...",
                "💭 让我想想怎么回答你...",
                "✨ 稍等一下，正在整理思路...",
                "🎀 思考中，请稍候...",
            ]
        elif "女仆" in role_name:
            placeholders = [
                "🎀 女仆正在为您准备回复...",
                "💝 请稍候，正在用心思考...",
                "🌹 为您整理答案中...",
                "✨ 恭敬地思考中...",
            ]
        elif "kei" in role_name or "Kei" in role_name:
            placeholders = [
                "⚡ Kei正在处理信息...",
                "🔥 分析中，稍等片刻...",
                "💪 正在组织语言...",
                "🎯 思考最佳回复方案...",
            ]
        elif "游戏" in role_name or "玩家" in role_name:
            placeholders = [
                "🎮 正在加载回复...",
                "🕹️ 思考攻略中...",
                "🎲 分析情况，请稍候...",
                "🏆 准备最佳策略...",
            ]
        elif "助手" in role_name or "AI" in role_name:
            placeholders = [
                "🤖 AI助手思考中...",
                "💻 正在处理您的请求...",
                "🔍 分析问题，准备回复...",
                "⚙️ 系统思考中...",
            ]
        else:
            # 默认占位信息
            placeholders = [
                f"💭 {role.name}正在思考...",
                f"✨ {role.name}准备回复中...",
                f"🌟 {role.name}整理思路中...",
            ]
        
        # 随机选择一个占位信息（简单轮换）
        import time
        index = int(time.time()) % len(placeholders)
        return placeholders[index]
    
    def _format_llm_response(self, text: str) -> str:
        """清理LLM返回的HTML内容，确保只包含Telegram支持的标签"""
        if not text:
            return text
        
        import re
        
        # 清理不支持的HTML标签
        # Telegram支持的HTML标签: <b>, <i>, <u>, <s>, <code>, <pre>, <a>
        unsupported_patterns = [
            r'</?dyn[^>]*>',     # 移除 <dyn> 标签
            r'</?span[^>]*>',    # 移除 <span> 标签
            r'</?div[^>]*>',     # 移除 <div> 标签
            r'</?p[^>]*>',       # 移除 <p> 标签
            r'</?strong[^>]*>',  # 移除 <strong> 标签（LLM应该用 <b>）
            r'</?em[^>]*>',      # 移除 <em> 标签（LLM应该用 <i>）
            r'</?h[1-6][^>]*>',  # 移除标题标签
            r'</?ul[^>]*>',      # 移除列表标签
            r'</?ol[^>]*>',      # 移除有序列表标签
            r'</?li[^>]*>',      # 移除列表项标签
            r'</?br[^>]*>',      # 移除换行标签
            r'</?hr[^>]*>',      # 移除分隔线标签
        ]
        
        # 先保护支持的标签
        supported_tags = ['b', 'i', 'u', 's', 'code', 'pre', 'a']
        protected_content = {}
        protect_counter = 0
        
        # 保护支持的标签
        for tag in supported_tags:
            pattern = f'<{tag}[^>]*>.*?</{tag}>'
            matches = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)
            for match in matches:
                placeholder = f"__PROTECTED_{protect_counter}__"
                protected_content[placeholder] = match
                text = text.replace(match, placeholder, 1)
                protect_counter += 1
        
        # 移除不支持的标签
        for pattern in unsupported_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # 恢复保护的标签
        for placeholder, original in protected_content.items():
            text = text.replace(placeholder, original)
        
        return text
    
    def _format_response(self, text: str, role_name: str = None) -> str:
        """格式化响应文本，添加Telegram富文本支持"""
        if not text:
            return text
        
        # 首先格式化LLM返回的内容
        text = self._format_llm_response(text)
        
        # 移除角色标识显示，让回复更自然
        # 角色信息已经在系统提示词中处理，不需要在用户看到的回复中显示
        

        
        return text
    
    async def _learn_from_conversation(self, user_id: int, conversation_id: int, messages) -> None:
        """从对话中学习知识"""
        try:
            # 检查是否启用自动学习
            from ..config import config
            if not config.rag.auto_learning:
                return
            
            # 检查消息数量是否达到学习触发条件
            if len(messages) < config.rag.learning_trigger_messages:
                return
            
            # 获取最近的消息内容
            recent_messages = messages[-config.rag.learning_trigger_messages:]
            message_contents = [msg.content for msg in recent_messages if msg.content and msg.content.strip()]
            
            if not message_contents:
                return
            
            # 提取知识
            knowledge_items = await self.knowledge_extractor.extract_from_conversation(
                messages=recent_messages,
                user_id=user_id,
                conversation_id=conversation_id
            )
            
            # 保存提取的知识
            saved_count = 0
            for item in knowledge_items:
                try:
                    # 创建知识条目对象
                    from ..services.rag_service import KnowledgeEntry
                    knowledge_entry = KnowledgeEntry(
                        user_id=item['user_id'],
                        conversation_id=item['conversation_id'],
                        title=item['title'],
                        content=item['content'],
                        summary=item.get('summary', item['title']),
                        keywords=item.get('keywords', ''),
                        category=item.get('category', 'conversation'),
                        importance_score=item.get('importance_score', 0.5)
                    )
                    
                    # 保存知识条目
                    entry_id = await self.rag_service._save_knowledge_entry(knowledge_entry)
                    if entry_id:
                        knowledge_entry.id = entry_id
                        # 生成向量嵌入
                        await self.rag_service._generate_embedding(knowledge_entry)
                        saved_count += 1
                        
                except Exception as e:
                    logger.warning(f"保存知识条目失败: {e}")
            
            if saved_count > 0:
                logger.info(f"从对话中学习并保存了 {saved_count} 个知识条目")
                
        except Exception as e:
            logger.warning(f"从对话中学习知识失败: {e}")
    
    async def _send_response(self, update, context, response_text: str, role_name: str = None):
        """发送格式化的响应"""
        try:
            # 格式化响应文本
            formatted_text = self._format_response(response_text, role_name)
            
            # 发送消息，启用HTML解析
            await update.message.reply_text(
                formatted_text,
                parse_mode='HTML',
                disable_web_page_preview=True
            )
            
        except Exception as e:
            logger.error(f"发送响应失败: {e}")
            # 如果HTML解析失败，发送纯文本
            try:
                await update.message.reply_text(response_text)
            except Exception as e2:
                logger.error(f"发送纯文本也失败: {e2}")
                await update.message.reply_text("抱歉，消息发送失败，请稍后再试。")
    
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
        """处理文本消息"""
        try:
            # 提取消息信息
            message_info = self.extract_message_info(update)
            if not message_info:
                logger.warning("无法提取消息信息或消息为空，跳过处理")
                return True  # 返回True，因为这不是错误，只是跳过空消息
                
            user_id = message_info["user_id"]
            chat_id = message_info["chat_id"]
            message = message_info["message"]
            
            # 记录用户到数据库
            conversation_id = None
            if self.database_service:
                try:
                    # 确保用户存在
                    db_user_id = await self.database_service.ensure_user(
                        telegram_user_id=user_id,
                        username=update.effective_user.username,
                        first_name=update.effective_user.first_name,
                        last_name=update.effective_user.last_name
                    )
                    
                    # 获取当前角色
                    current_role = self.conversation_service.get_role(user_id, chat_id)
                    role_name = current_role.name if current_role else "AI助手"
                    
                    # 确保对话存在
                    conversation_id = await self.database_service.ensure_conversation(
                        user_id=db_user_id,
                        chat_id=chat_id,
                        role_name=role_name
                    )
                    
                    # 保存用户消息
                    await self.database_service.save_message(conversation_id, message)
                    
                except Exception as e:
                    logger.warning(f"数据库记录失败: {e}")  # 不影响主流程
            
            # 获取或创建对话
            conversation = self.conversation_service.get_conversation(
                user_id=user_id,
                chat_id=chat_id
            )
            
            # 获取当前角色
            role = conversation.role
            if not role:
                # 如果没有设置角色，使用默认角色
                role = self.conversation_service.get_role(user_id, chat_id)
                if not role:
                    # 如果还是没有角色，创建一个默认角色
                    default_role_name = "AI助手"
                    self.conversation_service.set_role(user_id, chat_id, default_role_name)
                    role = self.conversation_service.get_role(user_id, chat_id)
            
            # 添加用户消息到对话
            self.conversation_service.add_message(
                user_id=user_id,
                chat_id=chat_id,
                message=message
            )
            
            # 先发送个性化占位信息
            placeholder_text = self._get_placeholder_message(role)
            placeholder_message = await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=placeholder_text,
                reply_to_message_id=update.message.message_id
            )
            
            try:
                # RAG检索相关知识
                retrieved_knowledge = []
                if self.rag_service and message.content:
                    try:
                        knowledge_results = await self.rag_service.retrieve_knowledge(
                            user_id=user_id,
                            query=message.content,
                            conversation_id=conversation_id
                        )
                        retrieved_knowledge = knowledge_results
                        if retrieved_knowledge:
                            logger.info(f"RAG检索到 {len(retrieved_knowledge)} 个相关知识")
                    except Exception as e:
                        logger.warning(f"RAG检索失败: {e}")
                
                # 构建系统提示词（包含RAG知识）
                system_prompt = self._build_system_prompt_with_rag(role, retrieved_knowledge)
                
                # 构建完整消息列表
                messages = [{"role": "system", "content": system_prompt}]
                
                # 添加对话历史（限制长度）
                for msg in conversation.messages[-10:]:  # 只保留最近10条消息
                    # 确保消息内容不为空
                    if msg.content and msg.content.strip():
                        messages.append({"role": msg.role, "content": msg.content.strip()})
                    else:
                        logger.warning(f"跳过空消息: role={msg.role}, content='{msg.content}'")
                
                # 获取可用的MCP工具
                tools = None
                if self.mcp_service:
                    tools = self.mcp_service.get_tools_for_llm()
                    if tools:
                        logger.info(f"可用的MCP工具数量: {len(tools)}")
                
                # 设置对话ID用于工具调用记录
                if conversation_id:
                    self.openai_service.set_conversation_id(conversation_id)
                
                # 调用OpenAI API（支持工具调用）
                response_text = await self.openai_service.chat_completion(messages, tools)
                
                # 验证响应
                if not response_text or not isinstance(response_text, str):
                    response_text = "抱歉，我无法生成有效的响应，请稍后重试。"
                
                # 格式化响应
                formatted_response = self._format_response(response_text)
                
                # 编辑占位信息，替换为实际响应
                try:
                    await context.bot.edit_message_text(
                        chat_id=update.effective_chat.id,
                        message_id=placeholder_message.message_id,
                        text=formatted_response,
                        parse_mode='HTML'
                    )
                except Exception as html_error:
                    # 如果HTML解析失败，尝试发送纯文本
                    logger.warning(f"HTML解析失败，回退到纯文本: {html_error}")
                    await context.bot.edit_message_text(
                        chat_id=update.effective_chat.id,
                        message_id=placeholder_message.message_id,
                        text=response_text  # 使用原始文本，不进行HTML格式化
                    )
                
                # 添加机器人响应到对话
                bot_message = Message(
                    role="assistant",
                    content=response_text,
                    timestamp=time.time()
                )
                self.conversation_service.add_message(
                    user_id=user_id,
                    chat_id=chat_id,
                    message=bot_message
                )
                
                # 保存AI响应到数据库
                if self.database_service and conversation_id:
                    try:
                        await self.database_service.save_message(conversation_id, bot_message)
                    except Exception as e:
                        logger.warning(f"保存AI响应失败: {e}")
                
                # RAG知识学习
                if self.rag_service and self.knowledge_extractor and conversation_id:
                    await self._learn_from_conversation(user_id, conversation_id, conversation.messages)
                
                logger.info(f"成功处理用户 {user_id} 的消息")
                return True
                
            except Exception as e:
                # 如果LLM处理失败，编辑占位信息为错误消息
                error_message = f"❌ 处理消息时出现错误：{str(e)}"
                await context.bot.edit_message_text(
                    chat_id=update.effective_chat.id,
                    message_id=placeholder_message.message_id,
                    text=error_message
                )
                logger.error(f"LLM处理失败: {e}")
                return False
                
        except Exception as e:
            logger.error(f"处理文本消息失败: {e}")
            # 尝试发送错误消息
            try:
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text="❌ 处理消息时出现错误，请稍后重试。",
                    reply_to_message_id=update.message.message_id
                )
            except Exception as send_error:
                logger.error(f"发送错误消息失败: {send_error}")
            return False
