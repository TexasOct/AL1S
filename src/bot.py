"""
Telegram机器人主类
"""
import asyncio
from typing import Optional
from telegram import Update, BotCommand
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes
from loguru import logger

from .config import config
from .services.openai_service import OpenAIService
from .services.ascii2d_service import Ascii2DService
from .services.conversation_service import ConversationService
from .services.mcp_service import MCPService
from .services.database_service import DatabaseService
from .services.rag_service import RAGService
from .utils.database_logger import init_database_logger
from .handlers.chat_handler import ChatHandler
from .handlers.command_handler import CommandHandler as CmdHandler
from .handlers.image_handler import ImageHandler


class AL1SBot:
    """AL1S-Bot主类"""
    
    def __init__(self):
        self.config = config
        self.application: Optional[Application] = None
        
        # 初始化MCP服务
        self.mcp_service = MCPService() if config.mcp.enabled else None
        
        # 初始化服务（传入MCP工具处理器）
        tool_handler = self._create_tool_handler() if self.mcp_service else None
        self.database_service = DatabaseService()
        
        # 初始化数据库记录器
        init_database_logger(self.database_service)
        
        self.openai_service = OpenAIService(tool_handler=tool_handler, database_service=self.database_service)
        self.ascii2d_service = Ascii2DService()
        self.conversation_service = ConversationService(database_service=self.database_service)
        
        # 初始化RAG服务
        self.rag_service = None
        if config.rag.enabled:
            try:
                self.rag_service = RAGService(
                    database_service=self.database_service,
                    vector_store_path=config.rag.vector_store_path
                )
                logger.info("RAG服务已启用")
            except Exception as e:
                logger.error(f"初始化RAG服务失败: {e}")
                logger.warning("RAG服务将不可用")
        
        # 初始化处理器
        self.chat_handler = ChatHandler(
            self.openai_service, 
            self.conversation_service, 
            self.mcp_service, 
            self.database_service,
            self.rag_service
        )
        self.image_handler = ImageHandler(self.ascii2d_service, self.openai_service, self.conversation_service)
        
        # 处理器列表
        self.handlers = [
            self.chat_handler,
            self.image_handler
        ]
        
        logger.info("AL1S-Bot 初始化完成")
    
    def _create_tool_handler(self):
        """创建MCP工具处理器"""
        async def tool_handler(tool_name: str, arguments: dict):
            """处理工具调用"""
            if self.mcp_service:
                return await self.mcp_service.call_tool(tool_name, arguments)
            return None
        return tool_handler
    
    async def _initialize_mcp_servers(self):
        """初始化MCP服务器"""
        if self.mcp_service and self.config.mcp.enabled:
            # 转换配置格式
            mcp_configs = []
            for server_config in self.config.mcp.servers:
                if server_config.enabled:
                    mcp_configs.append({
                        "name": server_config.name,
                        "command": server_config.command,
                        "args": server_config.args,
                        "env": server_config.env
                    })
            
            if mcp_configs:
                logger.info(f"正在初始化 {len(mcp_configs)} 个MCP服务器...")
                await self.mcp_service.initialize_default_servers(mcp_configs)
                
                # 显示已连接的工具
                tools = self.mcp_service.get_available_tools()
                if tools:
                    logger.info(f"已加载 {len(tools)} 个MCP工具: {list(tools.keys())}")
                else:
                    logger.warning("未找到任何可用的MCP工具")
            else:
                logger.info("没有启用的MCP服务器")
    
    async def _post_init_callback(self, application):
        """应用初始化后的回调，用于初始化MCP服务器和RAG服务"""
        if self.mcp_service:
            logger.info("正在初始化MCP服务...")
            await self._initialize_mcp_servers()
        
        # 初始化RAG服务
        if self.rag_service:
            logger.info("正在初始化RAG服务...")
            try:
                await self.rag_service.initialize()
                logger.info("RAG服务初始化完成")
            except Exception as e:
                logger.error(f"RAG服务初始化失败: {e}")
                self.rag_service = None
    
    def start(self):
        """启动机器人（支持优雅关闭的同步方法）"""
        try:
            # 创建应用
            self.application = (
                Application
                    .builder()
                    .token(self.config.telegram.bot_token)
                    .concurrent_updates(True)
                    .build()
            )
            
            # 设置处理器
            self._setup_handlers()
            
            # 设置错误处理器
            self.application.add_error_handler(self._error_handler)
            
            # 设置应用初始化后的回调
            if self.mcp_service:
                self.application.post_init = self._post_init_callback
            
            # 启动机器人
            logger.info("启动轮询模式...")
            
            # 使用标准的run_polling方法，MCP初始化将在post_init中进行
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
            self.image_handler,  # 传入图片处理器引用
            self.mcp_service,    # 传入MCP服务引用
            self.database_service # 传入数据库服务引用
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
        self.application.add_handler(
            CommandHandler("tools", self._handle_tools_command)
        )
        self.application.add_handler(
            CommandHandler("mcp_status", self._handle_mcp_status_command)
        )
        self.application.add_handler(
            CommandHandler("db_stats", self._handle_db_stats_command)
        )
        self.application.add_handler(
            CommandHandler("my_stats", self._handle_my_stats_command)
        )
        self.application.add_handler(
            CommandHandler("rag_stats", self._handle_rag_stats_command)
        )
        self.application.add_handler(
            CommandHandler("knowledge", self._handle_knowledge_command)
        )
        self.application.add_handler(
            CommandHandler("learn", self._handle_learn_command)
        )
        self.application.add_handler(
            CommandHandler("forget", self._handle_forget_command)
        )
        self.application.add_handler(
            CommandHandler("rebuild_index", self._handle_rebuild_index_command)
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
                BotCommand("test_search", "测试图片搜索服务"),
                BotCommand("tools", "显示可用的MCP工具"),
                BotCommand("mcp_status", "显示MCP服务器状态"),
                BotCommand("db_stats", "显示数据库统计信息"),
                BotCommand("my_stats", "显示我的使用统计"),
                BotCommand("rag_stats", "显示RAG知识库统计"),
                BotCommand("knowledge", "管理知识库"),
                BotCommand("learn", "从当前对话学习知识"),
                BotCommand("forget", "清理旧知识"),
                BotCommand("rebuild_index", "重建向量索引")
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
    
    async def _handle_tools_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理tools命令"""
        await self.command_handler._handle_tools(update, context)
    
    async def _handle_mcp_status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理mcp_status命令"""
        await self.command_handler._handle_mcp_status(update, context)
    
    async def _handle_db_stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理db_stats命令"""
        await self.command_handler._handle_db_stats(update, context)
    
    async def _handle_my_stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理my_stats命令"""
        await self.command_handler._handle_my_stats(update, context)
    
    async def _handle_rag_stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理rag_stats命令"""
        try:
            if not self.rag_service:
                await update.message.reply_text("❌ RAG服务未启用")
                return
            
            stats = await self.rag_service.get_rag_stats()
            
            message = "📊 <b>RAG知识库统计</b>\n\n"
            
            for table_name, info in stats.items():
                if table_name == 'vector_index':
                    message += f"🔍 <b>向量索引</b>\n"
                    message += f"• 向量数量: {info['total_vectors']}\n"
                    message += f"• 向量维度: {info['dimension']}\n"
                    message += f"• 索引类型: {info['index_type']}\n\n"
                else:
                    display_name = {
                        'knowledge_entries': '知识条目',
                        'embeddings': '向量嵌入',
                        'knowledge_retrievals': '检索记录'
                    }.get(table_name, table_name)
                    
                    message += f"📚 <b>{display_name}</b>\n"
                    message += f"• 总数量: {info['total_count']}\n"
                    message += f"• 用户数: {info.get('unique_users', 0)}\n"
                    
                    metric_name = "平均重要性" if table_name == 'knowledge_entries' else \
                                 "平均维度" if table_name == 'embeddings' else "使用率"
                    message += f"• {metric_name}: {info['additional_metric']:.2f}\n"
                    
                    if info['last_created']:
                        message += f"• 最后更新: {info['last_created']}\n"
                    message += "\n"
            
            await update.message.reply_text(message, parse_mode='HTML')
            
        except Exception as e:
            logger.error(f"获取RAG统计失败: {e}")
            await update.message.reply_text("❌ 获取RAG统计失败")
    
    async def _handle_knowledge_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理knowledge命令"""
        try:
            if not self.rag_service:
                await update.message.reply_text("❌ RAG服务未启用")
                return
            
            user_id = update.effective_user.id
            args = context.args
            
            if not args:
                await update.message.reply_text(
                    "📚 <b>知识库管理</b>\n\n"
                    "使用方法:\n"
                    "• <code>/knowledge search 关键词</code> - 搜索知识\n"
                    "• <code>/knowledge list</code> - 列出我的知识\n"
                    "• <code>/knowledge add 标题 内容</code> - 添加知识\n",
                    parse_mode='HTML'
                )
                return
            
            command = args[0].lower()
            
            if command == "search" and len(args) > 1:
                query = " ".join(args[1:])
                results = await self.rag_service.retrieve_knowledge(user_id, query)
                
                if not results:
                    await update.message.reply_text("🔍 没有找到相关知识")
                    return
                
                message = f"🔍 <b>搜索结果：{query}</b>\n\n"
                for i, (entry, score) in enumerate(results[:5], 1):
                    message += f"{i}. <b>{entry.title}</b>\n"
                    message += f"内容: {entry.content[:100]}...\n"
                    message += f"相关性: {score:.2f}\n\n"
                
                await update.message.reply_text(message, parse_mode='HTML')
                
            elif command == "add" and len(args) > 2:
                title = args[1]
                content = " ".join(args[2:])
                
                knowledge_item = await self.chat_handler.knowledge_extractor.extract_single_knowledge(
                    content=f"{title}: {content}",
                    user_id=user_id,
                    conversation_id=None,
                    category="manual"
                )
                
                if knowledge_item:
                    from .services.rag_service import KnowledgeEntry
                    entry = KnowledgeEntry(**knowledge_item)
                    entry_id = await self.rag_service._save_knowledge_entry(entry)
                    
                    if entry_id:
                        entry.id = entry_id
                        await self.rag_service._generate_embedding(entry)
                        await update.message.reply_text("✅ 知识已添加到知识库")
                    else:
                        await update.message.reply_text("❌ 添加知识失败")
                else:
                    await update.message.reply_text("❌ 知识内容无效")
            
            else:
                await update.message.reply_text("❌ 无效的命令格式")
                
        except Exception as e:
            logger.error(f"处理知识命令失败: {e}")
            await update.message.reply_text("❌ 处理命令失败")
    
    async def _handle_learn_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理learn命令 - 手动触发学习"""
        try:
            if not self.rag_service or not self.chat_handler.knowledge_extractor:
                await update.message.reply_text("❌ RAG服务未启用")
                return
            
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            
            # 获取当前对话
            conversation = self.conversation_service.get_conversation(user_id, chat_id)
            if not conversation.messages:
                await update.message.reply_text("❌ 没有对话历史可供学习")
                return
            
            # 提取知识
            knowledge_items = await self.chat_handler.knowledge_extractor.extract_from_conversation(
                messages=conversation.messages,
                user_id=user_id,
                conversation_id=None
            )
            
            if not knowledge_items:
                await update.message.reply_text("📚 没有从当前对话中提取到新知识")
                return
            
            # 保存知识
            saved_count = 0
            for item in knowledge_items:
                try:
                    from .services.rag_service import KnowledgeEntry
                    entry = KnowledgeEntry(**item)
                    entry_id = await self.rag_service._save_knowledge_entry(entry)
                    
                    if entry_id:
                        entry.id = entry_id
                        await self.rag_service._generate_embedding(entry)
                        saved_count += 1
                        
                except Exception as e:
                    logger.warning(f"保存知识失败: {e}")
            
            await update.message.reply_text(f"✅ 从对话中学习并保存了 {saved_count} 个知识条目")
            
        except Exception as e:
            logger.error(f"处理学习命令失败: {e}")
            await update.message.reply_text("❌ 学习失败")
    
    async def _handle_forget_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理forget命令 - 清理旧知识"""
        try:
            if not self.rag_service:
                await update.message.reply_text("❌ RAG服务未启用")
                return
            
            args = context.args
            days = 90  # 默认清理90天前的知识
            
            if args and args[0].isdigit():
                days = int(args[0])
            
            deleted_count = await self.rag_service.cleanup_old_knowledge(days)
            
            await update.message.reply_text(f"🧹 清理了 {deleted_count} 个旧知识条目（{days}天前）")
            
        except Exception as e:
            logger.error(f"处理遗忘命令失败: {e}")
            await update.message.reply_text("❌ 清理失败")
    
    async def _handle_rebuild_index_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理rebuild_index命令 - 重建向量索引"""
        try:
            if not self.rag_service:
                await update.message.reply_text("❌ RAG服务未启用")
                return
            
            await update.message.reply_text("🔄 正在重建向量索引...")
            
            success = await self.rag_service.rebuild_index()
            
            if success:
                await update.message.reply_text("✅ 向量索引重建完成")
            else:
                await update.message.reply_text("❌ 向量索引重建失败")
                
        except Exception as e:
            logger.error(f"重建索引失败: {e}")
            await update.message.reply_text("❌ 重建索引失败")
    
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
            
            # 清理MCP服务
            if hasattr(self, 'mcp_service') and self.mcp_service:
                logger.info("正在清理MCP服务...")
                try:
                    asyncio.run(self.mcp_service.close_all())
                    logger.info("MCP服务清理完成")
                except Exception as mcp_error:
                    logger.error(f"清理MCP服务失败: {mcp_error}")
                    
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
