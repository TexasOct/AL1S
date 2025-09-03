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
from ..models import Message


class ChatHandler(BaseHandler):
    """聊天处理器"""
    
    def __init__(self, openai_service: OpenAIService, conversation_service: ConversationService):
        super().__init__("ChatHandler", "处理用户聊天消息")
        self.openai_service = openai_service
        self.conversation_service = conversation_service
    
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
        if role and hasattr(role, 'personality'):
            return f"你是{role.name}，{role.personality}。请用这个角色来回复用户。"
        return "你是一个有用的AI助手。"
    
    def _format_llm_response(self, text: str) -> str:
        """格式化LLM返回的内容，使其符合Telegram API实体规范"""
        if not text:
            return text
        
        import re
        
        # 1. 格式化代码块 - 使用 <pre> 标签
        # 匹配 ```language\ncode``` 或 ```code``` 格式
        text = re.sub(r'```(\w+)?\n(.*?)```', r'<pre>\2</pre>', text, flags=re.DOTALL)
        
        # 2. 格式化行内代码 - 使用 <code> 标签
        # 匹配 `code` 格式，但排除已经处理的代码块
        # 使用更简单的方法，先处理代码块，再处理行内代码
        text = re.sub(r'`([^`\n]+)`', r'<code>\1</code>', text)
        
        # 3. 格式化粗体 - 使用 <b> 标签
        # 匹配 **text** 格式
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        
        # 4. 格式化斜体 - 使用 <i> 标签
        # 匹配 *text* 格式，但排除粗体标记
        # 使用更简单的方法，避免复杂的后向查找
        text = re.sub(r'(?<!\*)\*([^*\n]+)\*(?!\*)', r'<i>\1</i>', text)
        
        # 5. 格式化删除线 - 使用 <s> 标签
        # 匹配 ~~text~~ 格式
        text = re.sub(r'~~(.*?)~~', r'<s>\1</s>', text)
        
        # 6. 格式化列表
        lines = text.split('\n')
        formatted_lines = []
        in_list = False
        
        for line in lines:
            stripped = line.strip()
            
            # 无序列表
            if stripped.startswith('- '):
                if not in_list:
                    in_list = True
                line = f"• {stripped[2:]}"
            # 有序列表
            elif re.match(r'^\d+\.\s', stripped):
                if not in_list:
                    in_list = True
                line = f"<b>{stripped}</b>"
            # 空行或非列表项
            elif stripped == '':
                in_list = False
            else:
                in_list = False
            
            formatted_lines.append(line)
        
        text = '\n'.join(formatted_lines)
        
        # 7. 格式化标题（如果LLM返回了Markdown标题）
        # 匹配 # 标题 格式
        text = re.sub(r'^#\s+(.+)$', r'<b>📋 \1</b>', text, flags=re.MULTILINE)
        text = re.sub(r'^##\s+(.+)$', r'<b>📌 \1</b>', text, flags=re.MULTILINE)
        text = re.sub(r'^###\s+(.+)$', r'<b>📍 \1</b>', text, flags=re.MULTILINE)
        
        # 8. 格式化引用块
        # 匹配 > 引用 格式
        text = re.sub(r'^>\s+(.+)$', r'<i>💬 \1</i>', text, flags=re.MULTILINE)
        
        # 9. 格式化链接（如果LLM返回了Markdown链接）
        # 匹配 [text](url) 格式
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)
        
        return text
    
    def _format_response(self, text: str, role_name: str = None) -> str:
        """格式化响应文本，添加Telegram富文本支持"""
        if not text:
            return text
        
        # 首先格式化LLM返回的内容
        text = self._format_llm_response(text)
        
        # 添加角色标识
        if role_name:
            text = f"<b>🤖 {role_name}</b>\n\n{text}"
        
        # # 添加分隔线
        # if role_name:
        #     text += "\n\n" + "─" * 30
        
        return text
    
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
    
    async def handle(self, update, context):
        """处理文本消息"""
        try:
            # 记录处理开始
            self.log_handling(update, "文本消息")
            
            # 获取用户信息
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            message_text = update.message.text
            
            # 获取或创建对话（注意：这些方法不是异步的）
            conversation = self.conversation_service.get_conversation(user_id, chat_id)
            current_role = self.conversation_service.get_role(user_id, chat_id)
            
            # 记录用户消息
            user_message = Message(
                role="user",
                content=message_text,
                timestamp=update.message.date.timestamp()
            )
            self.conversation_service.add_message(user_id, chat_id, user_message)
            
            # 构建系统提示词（注意：这个方法现在是异步的）
            system_prompt = self._build_system_prompt(current_role)
            
            # 获取对话历史
            messages = conversation.messages if hasattr(conversation, 'messages') else []
            
            # 构建OpenAI请求消息
            openai_messages = [{"role": "system", "content": system_prompt}]
            
            # 添加对话历史（限制长度）
            for msg in messages[-10:]:  # 只保留最近10条消息
                openai_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
            
            # 调用OpenAI API
            response_text = await self.openai_service.chat_completion(openai_messages)
            
            # 检查响应是否有效
            if not response_text or not isinstance(response_text, str):
                logger.error(f"OpenAI API 返回无效响应: {type(response_text)} = {response_text}")
                response_text = "抱歉，我无法生成有效的回复，请稍后再试。"
            
            # 记录AI响应
            ai_message = Message(
                role="assistant",
                content=response_text,
                timestamp=update.message.date.timestamp()
            )
            self.conversation_service.add_message(user_id, chat_id, ai_message)
            
            # 发送格式化的响应
            role_name = current_role.name if current_role else None
            await self._send_response(update, context, response_text, role_name)
            
        except Exception as e:
            logger.error(f"处理文本消息失败: {e}")
            await update.message.reply_text("抱歉，处理消息时出现错误，请稍后再试。")
