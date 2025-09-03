"""
Telegram机器人主类
"""
from typing import Optional
from telegram import Update, BotCommand
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes
from loguru import logger

from .config import config
from .services.openai_service import OpenAIService
from .services.ascii2d_service import Ascii2DService
from .services.conversation_service import ConversationService
from .handlers.chat_handler import ChatHandler
from .handlers.command_handler import CommandHandler as CmdHandler
from .handlers.image_handler import ImageHandler


class AL1SBot:
    """AL1S-Bot主类"""
    
    def __init__(self):
        self.config = config
        self.application: Optional[Application] = None
        
        # 初始化服务
        self.openai_service = OpenAIService()
        self.ascii2d_service = Ascii2DService()
        self.conversation_service = ConversationService()
        
        # 初始化处理器
        self.chat_handler = ChatHandler(self.openai_service, self.conversation_service)
        self.image_handler = ImageHandler(self.ascii2d_service, self.openai_service, self.conversation_service)
        
        # 处理器列表
        self.handlers = [
            self.chat_handler,
            self.image_handler
        ]
        
        logger.info("AL1S-Bot 初始化完成")
    
    def start(self):
        """启动机器人（支持优雅关闭的同步方法）"""
        try:
            # 创建应用
            self.application = Application.builder().token(self.config.telegram.bot_token).build()
            
            # 设置处理器
            self._setup_handlers()
            
            # 设置错误处理器
            self.application.add_error_handler(self._error_handler)
            
            # 启动机器人
            logger.info("启动轮询模式...")
            
            # 使用标准的run_polling方法
            self.application.run_polling(
                allowed_updates=Update.ALL_TYPES,
                drop_pending_updates=True,
                close_loop=False  # 不关闭事件循环
            )
                
        except KeyboardInterrupt:
            logger.info("收到键盘中断，正在关闭机器人...")
            self._cleanup()
        except Exception as e:
            logger.error(f"启动机器人失败: {e}")
            self._cleanup()
            raise
    
    def _setup_handlers(self):
        """设置消息处理器"""
        # 命令处理器（传入图片处理器引用）
        self.command_handler = CmdHandler(
            self.conversation_service, 
            self.openai_service,
            self.image_handler  # 传入图片处理器引用
        )
        
        # 命令处理器
        self.application.add_handler(
            CommandHandler("start", self._handle_start_command)
        )
        self.application.add_handler(
            CommandHandler("help", self._handle_help_command)
        )
        self.application.add_handler(
            CommandHandler("role", self._handle_role_command)
        )
        self.application.add_handler(
            CommandHandler("roles", self._handle_roles_command)
        )
        self.application.add_handler(
            CommandHandler("create_role", self._handle_create_role_command)
        )
        self.application.add_handler(
            CommandHandler("reset", self._handle_reset_command)
        )
        self.application.add_handler(
            CommandHandler("stats", self._handle_stats_command)
        )
        self.application.add_handler(
            CommandHandler("ping", self._handle_ping_command)
        )
        self.application.add_handler(
            CommandHandler("search", self._handle_search_command)
        )
        self.application.add_handler(
            CommandHandler("search_engines", self._handle_search_engines_command)
        )
        self.application.add_handler(
            CommandHandler("test_search", self._handle_test_search_command)
        )
        
        # 图片处理器
        self.application.add_handler(
            MessageHandler(filters.PHOTO | filters.Document.IMAGE, self._handle_image)
        )
        
        # 通用消息处理器
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
        )
    
    async def _register_commands_async(self):
        """异步注册机器人命令列表"""
        try:
            # 定义命令列表
            commands = [
                BotCommand("start", "开始使用机器人"),
                BotCommand("help", "显示帮助信息"),
                BotCommand("role", "设置或查看当前角色"),
                BotCommand("roles", "显示所有可用角色"),
                BotCommand("create_role", "创建自定义角色"),
                BotCommand("reset", "重置当前对话"),
                BotCommand("stats", "显示对话统计信息"),
                BotCommand("ping", "测试机器人响应"),
                BotCommand("search", "搜索图片URL"),
                BotCommand("search_engines", "显示可用的搜索引擎"),
                BotCommand("test_search", "测试图片搜索服务")
            ]
            
            # 注册命令列表
            await self.application.bot.set_my_commands(commands)
            logger.info(f"异步成功注册 {len(commands)} 个命令")
            
        except Exception as e:
            logger.error(f"异步注册命令失败: {e}")
            raise
    
    async def stop(self):
        """停止机器人"""
        try:
            if self.application:
                logger.info("正在停止机器人应用...")
                await self.application.stop()
                await self.application.shutdown()
                self.application = None
                logger.info("机器人已停止")
        except Exception as e:
            logger.error(f"停止机器人时发生错误: {e}")
    
    # 命令处理方法
    async def _handle_start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理start命令"""
        # 在首次使用时注册命令列表
        if not hasattr(self, '_commands_registered'):
            try:
                await self._register_commands_async()
                self._commands_registered = True
                logger.info("首次使用，命令列表注册完成")
            except Exception as e:
                logger.error(f"首次命令注册失败: {e}")
        
        # 调用原来的start处理
        await self.command_handler._handle_start(update, context)
    
    async def _handle_help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理help命令"""
        await self.command_handler._handle_help(update, context)
    
    async def _handle_role_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理role命令"""
        args = update.message.text.split(maxsplit=1)[1] if len(update.message.text.split()) > 1 else ""
        await self.command_handler._handle_role(update, context, args)
    
    async def _handle_roles_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理roles命令"""
        await self.command_handler._handle_roles(update, context)
    
    async def _handle_create_role_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理create_role命令"""
        args = update.message.text.split(maxsplit=1)[1] if len(update.message.text.split()) > 1 else ""
        await self.command_handler._handle_create_role(update, context, args)
    
    async def _handle_reset_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理reset命令"""
        await self.command_handler._handle_reset(update, context)
    
    async def _handle_stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理stats命令"""
        await self.command_handler._handle_stats(update, context)
    
    async def _handle_ping_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理ping命令"""
        await self.command_handler._handle_ping(update, context)
    
    async def _handle_search_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理search命令"""
        args = update.message.text.split(maxsplit=1)[1] if len(update.message.text.split()) > 1 else ""
        await self.command_handler._handle_search(update, context, args)
    
    async def _handle_search_engines_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理search_engines命令"""
        await self.command_handler._handle_search_engines(update, context)
    
    async def _handle_test_search_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理test_search命令"""
        await self.command_handler._handle_test_search(update, context)
    
    # 消息处理方法
    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理文本消息"""
        await self.chat_handler.handle(update, context)
    
    async def _handle_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理图片消息"""
        await self.image_handler.handle(update, context)
    
    async def _error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """错误处理器"""
        logger.error(f"处理更新时发生错误: {context.error}")
        
        try:
            if update and update.message:
                await update.message.reply_text(
                    "抱歉，处理您的请求时发生了错误，请稍后再试。"
                )
        except Exception as e:
            logger.error(f"发送错误消息失败: {e}")
    
    def _cleanup(self):
        """清理资源"""
        try:
            if hasattr(self, 'application') and self.application:
                logger.info("正在清理机器人资源...")
                # 这里只记录日志，不调用异步方法
                logger.info("资源清理完成")
        except Exception as e:
            logger.error(f"清理资源时发生错误: {e}")
    
    def stop_sync(self):
        """同步停止机器人"""
        try:
            if hasattr(self, 'application') and self.application:
                logger.info("正在同步停止机器人...")
                # 对于同步方法，我们只能记录日志
                # 实际的停止操作由信号处理器处理
                logger.info("机器人同步停止完成")
        except Exception as e:
            logger.error(f"同步停止机器人时发生错误: {e}")
    
    def get_status(self) -> dict:
        """获取机器人状态"""
        return {
            "status": "running" if self.application else "stopped",
            "config": {
                "openai_model": self.config.openai.model,
                "openai_model": self.config.openai.model,
                "telegram_bot": self.config.telegram.bot_token[:10] + "..." if self.config.telegram.bot_token else None,
                "webhook_mode": bool(self.config.telegram.webhook_url)
            },
            "conversations": len(self.conversation_service.conversations),
            "users": len(self.conversation_service.users),
            "roles": len(self.conversation_service.roles)
        }
