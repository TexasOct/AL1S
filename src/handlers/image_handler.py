"""
图片处理器模块
"""
import io
from typing import Optional, List
from telegram import Update
from telegram.ext import ContextTypes
from loguru import logger

from .base_handler import BaseHandler
from ..services.ascii2d_service import Ascii2DService
from ..services.openai_service import OpenAIService
from ..services.conversation_service import ConversationService
from ..models import Message, ImageSearchResult


class ImageHandler(BaseHandler):
    """图片处理器"""
    
    def __init__(self, ascii2d_service: Ascii2DService, openai_service: OpenAIService, conversation_service: ConversationService):
        super().__init__("ImageHandler", "处理图片消息和图片搜索")
        self.ascii2d_service = ascii2d_service
        self.openai_service = openai_service
        self.conversation_service = conversation_service
    
    def can_handle(self, update: Update) -> bool:
        """检查是否可以处理此更新"""
        return (
            update.message is not None and
            (update.message.photo or update.message.document)
        )
    
    def _format_image_analysis(self, analysis_text: str, search_results: List[ImageSearchResult] = None) -> str:
        """格式化图片分析结果，添加Telegram富文本支持"""
        if not analysis_text:
            return "无法分析图片内容"
        
        # 构建响应文本
        response_text = "<b>🖼️ 图片分析结果</b>\n\n"
        response_text += f"<i>{analysis_text}</i>\n\n"
        
        # 添加图片搜索结果
        if search_results and len(search_results) > 0:
            response_text += "<b>🔍 相似图片搜索结果</b>\n\n"
            
            for i, result in enumerate(search_results[:5], 1):  # 只显示前5个结果
                response_text += f"<b>{i}. {result.title or '无标题'}</b>\n"
                
                if result.url:
                    response_text += f"   🔗 <a href='{result.url}'>查看图片</a>\n"
                
                if result.source and result.source != "Ascii2D":
                    response_text += f"   📍 来源: {result.source}\n"
                
                if result.metadata and result.metadata.get('author'):
                    response_text += f"   👤 作者: {result.metadata['author']}\n"
                
                response_text += "\n"
            
            if len(search_results) > 5:
                response_text += f"<i>... 还有 {len(search_results) - 5} 个结果</i>\n"
        else:
            response_text += "<i>⚠️ 未找到相似的图片搜索结果</i>\n"
        
        # 添加分隔线
        response_text += "\n" + "─" * 30
        
        return response_text
    
    async def _send_formatted_response(self, update, text: str):
        """发送格式化的响应"""
        try:
            # 发送消息，启用HTML解析
            await update.message.reply_text(
                text,
                parse_mode='HTML',
                disable_web_page_preview=True
            )
            
        except Exception as e:
            logger.error(f"发送格式化响应失败: {e}")
            # 如果HTML解析失败，发送纯文本
            try:
                # 移除HTML标签
                import re
                clean_text = re.sub(r'<[^>]+>', '', text)
                await update.message.reply_text(clean_text)
            except Exception as e2:
                logger.error(f"发送纯文本也失败: {e2}")
                await update.message.reply_text("抱歉，消息发送失败，请稍后再试。")
    
    async def handle(self, update, context):
        """处理图片消息"""
        try:
            # 记录处理开始
            self.log_handling(update, "图片消息")
            
            # 获取用户信息
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            
            # 获取图片文件
            photo = update.message.photo[-1]  # 获取最大尺寸的图片
            photo_file = await context.bot.get_file(photo.file_id)
            
            # 下载图片数据
            image_data = await photo_file.download_as_bytearray()
            image_bytes = bytes(image_data)
            
            # 记录用户消息
            user_message = Message(
                role="user",
                content="[图片]",
                timestamp=update.message.date.timestamp()
            )
            await self.conversation_service.add_message(user_id, chat_id, user_message)
            
            # 分析图片内容
            analysis_text = await self.openai_service.analyze_image(
                photo_file.file_path,
                "请描述这张图片的内容，包括主要元素、风格、可能的来源等信息"
            )
            
            # 搜索相似图片
            search_results = await self.ascii2d_service.search_by_image_file(image_bytes)
            
            # 格式化响应
            formatted_response = self._format_image_analysis(analysis_text, search_results)
            
            # 发送响应
            await self._send_formatted_response(update, formatted_response)
            
            # 记录AI响应
            ai_message = Message(
                role="assistant",
                content=formatted_response,
                timestamp=update.message.date.timestamp()
            )
            await self.conversation_service.add_message(user_id, chat_id, ai_message)
            
        except Exception as e:
            logger.error(f"处理图片消息失败: {e}")
            await update.message.reply_text("抱歉，处理图片时出现错误，请稍后再试。")
    
    async def _get_photo_file(self, update: Update):
        """获取图片文件"""
        try:
            if update.message.photo:
                # 获取最大尺寸的图片
                photo = max(update.message.photo, key=lambda p: p.file_size)
                return await update.get_bot().get_file(photo.file_id)
            elif update.message.document and update.message.document.mime_type.startswith('image/'):
                return await update.get_bot().get_file(update.message.document.file_id)
            else:
                return None
        except Exception as e:
            logger.error(f"获取图片文件失败: {e}")
            return None
    
    async def _analyze_image(self, photo_file) -> str:
        """分析图片内容"""
        try:
            # 获取图片URL
            file_url = photo_file.file_path
            if not file_url.startswith('http'):
                file_url = f"https://api.telegram.org/file/bot{photo_file.bot.token}/{file_url}"
            
            # 使用OpenAI分析图片
            analysis = await self.openai_service.analyze_image(
                file_url,
                "请简要描述这张图片的内容和风格"
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"图片分析失败: {e}")
            return "无法分析图片内容"
    
    async def _search_similar_images(self, photo_file) -> List[ImageSearchResult]:
        """搜索相似图片"""
        try:
            # 下载图片数据
            image_data = await photo_file.download_as_bytearray()
            
            # 将bytearray转换为bytes
            image_bytes = bytes(image_data)
            
            # 使用Ascii2D搜索
            results = await self.ascii2d_service.search_by_image_file(image_bytes)
            
            return results
            
        except Exception as e:
            logger.error(f"图片搜索失败: {e}")
            return []
    
    def _build_image_reply(self, analysis: str, search_results: List[ImageSearchResult]) -> str:
        """构建图片回复消息"""
        reply_parts = []
        
        # 添加图片分析结果
        reply_parts.append("🔍 图片分析结果：")
        reply_parts.append(analysis)
        reply_parts.append("")
        
        # 添加搜索结果
        if search_results:
            reply_parts.append("🖼️ 相似图片搜索结果：")
            
            for i, result in enumerate(search_results[:5], 1):  # 只显示前5个结果
                source_info = f"来源: {result.source}"
                if result.similarity:
                    source_info += f" (相似度: {result.similarity:.1f}%)"
                
                reply_parts.append(f"{i}. {source_info}")
                if result.title:
                    reply_parts.append(f"   标题: {result.title}")
                reply_parts.append(f"   链接: {result.url}")
                reply_parts.append("")
            
            if len(search_results) > 5:
                reply_parts.append(f"... 还有 {len(search_results) - 5} 个结果")
        else:
            reply_parts.append("❌ 未找到相似图片")
        
        return "\n".join(reply_parts)
    
    async def handle_image_search_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE, image_url: str) -> bool:
        """处理图片搜索命令"""
        try:
            # 验证URL
            if not self.ascii2d_service.validate_image_url(image_url):
                await update.message.reply_text("❌ 无效的图片URL")
                return False
            
            # 发送"正在搜索"状态
            await context.bot.send_chat_action(
                chat_id=update.message.chat_id,
                action="typing"
            )
            
            # 搜索相似图片（不再需要异步上下文管理器）
            search_results = await self.ascii2d_service.search_by_image_url(image_url)
            
            # 构建回复
            if search_results:
                reply_text = self._build_image_reply("通过URL搜索的图片", search_results)
            else:
                reply_text = "❌ 未找到相似图片"
            
            await update.message.reply_text(reply_text)
            return True
            
        except Exception as e:
            logger.error(f"URL图片搜索失败: {e}")
            await update.message.reply_text("抱歉，图片搜索失败，请稍后再试。")
            return False
    
    async def handle_multiple_engine_search(self, update: Update, context: ContextTypes.DEFAULT_TYPE, image_data: bytes, engines: List[str] = None) -> bool:
        """处理多引擎图片搜索"""
        try:
            # 发送"正在搜索"状态
            await context.bot.send_chat_action(
                chat_id=update.message.chat_id,
                action="typing"
            )
            
            # 使用多个搜索引擎搜索
            all_results = await self.ascii2d_service.search_multiple_engines(image_data, engines)
            
            # 构建多引擎搜索结果回复
            reply_text = self._build_multi_engine_reply(all_results)
            
            await update.message.reply_text(reply_text)
            return True
            
        except Exception as e:
            logger.error(f"多引擎图片搜索失败: {e}")
            await update.message.reply_text("抱歉，多引擎搜索失败，请稍后再试。")
            return False
    
    def _build_multi_engine_reply(self, all_results: dict) -> str:
        """构建多引擎搜索结果回复"""
        reply_parts = []
        reply_parts.append("🔍 多引擎图片搜索结果：")
        reply_parts.append("")
        
        total_results = 0
        
        for engine_name, results in all_results.items():
            if results:
                reply_parts.append(f"📊 {engine_name.upper()} 引擎：")
                reply_parts.append(f"   找到 {len(results)} 个结果")
                
                # 显示前3个结果
                for i, result in enumerate(results[:3], 1):
                    source_info = f"来源: {result.source}"
                    if result.similarity:
                        source_info += f" (相似度: {result.similarity:.1f}%)"
                    
                    reply_parts.append(f"   {i}. {source_info}")
                    if result.title:
                        reply_parts.append(f"      标题: {result.title}")
                    reply_parts.append(f"      链接: {result.url}")
                
                if len(results) > 3:
                    reply_parts.append(f"   ... 还有 {len(results) - 3} 个结果")
                
                reply_parts.append("")
                total_results += len(results)
            else:
                reply_parts.append(f"❌ {engine_name.upper()} 引擎：未找到结果")
                reply_parts.append("")
        
        reply_parts.append(f"📈 总计找到 {total_results} 个结果")
        
        return "\n".join(reply_parts)
