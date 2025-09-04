"""
对话管理服务
"""
import time
from typing import Dict, List, Optional, Any
from loguru import logger

from ..config import config
from ..models import Conversation, Message, Role


class ConversationService:
    """对话管理服务"""
    
    def __init__(self, database_service=None):
        self.conversations: Dict[str, Conversation] = {}
        self.users: Dict[int, Dict[str, Any]] = {}
        self.roles: Dict[str, Role] = {}
        self.database_service = database_service
        self._initialize_roles()
        logger.info("对话服务初始化完成")
    
    def _initialize_roles(self):
        """初始化角色配置"""
        try:
            # 从配置文件加载角色
            for role_name, role_config in config.roles.items():
                role = Role(
                    name=role_config.name,
                    english_name=role_config.english_name,
                    description=role_config.description,
                    personality=role_config.personality,
                    greeting=role_config.greeting,
                    farewell=role_config.farewell
                )
                self.roles[role_name] = role
                logger.info(f"加载角色: {role_name}")
            
            logger.info(f"成功加载 {len(self.roles)} 个角色")
        except Exception as e:
            logger.error(f"初始化角色失败: {e}")
    
    def get_conversation(self, user_id: int, chat_id: int) -> Conversation:
        """获取或创建对话"""
        key = f"{user_id}_{chat_id}"
        if key not in self.conversations:
            # 创建新对话，使用默认角色
            default_role = config.get_default_role()
            if default_role:
                role = Role(
                    name=default_role.name,
                    english_name=default_role.english_name,
                    description=default_role.description,
                    personality=default_role.personality,
                    greeting=default_role.greeting,
                    farewell=default_role.farewell
                )
            else:
                # 如果没有默认角色，创建一个基础角色
                role = Role(
                    name="AI助手",
                    english_name="AI Assistant",
                    description="智能AI助手",
                    personality="你是一个智能、友好的AI助手，能够帮助用户解决各种问题。",
                    greeting="您好！我是AI助手，很高兴为您服务。",
                    farewell="感谢您的使用！如果还有其他问题，随时可以找我。"
                )
            
            self.conversations[key] = Conversation(
                user_id=user_id,
                chat_id=chat_id,
                role=role,
                messages=[],
                created_at=time.time(),
                last_activity=time.time()
            )
            logger.info(f"为用户 {user_id} 在聊天 {chat_id} 创建新对话")
        
        return self.conversations[key]
    
    def add_message(self, user_id: int, chat_id: int, message: Message) -> None:
        """添加消息到对话"""
        conversation = self.get_conversation(user_id, chat_id)
        conversation.messages.append(message)
        conversation.last_activity = time.time()
        
        # 限制消息数量，保持对话长度合理
        max_messages = 50
        if len(conversation.messages) > max_messages:
            # 保留最新的消息
            conversation.messages = conversation.messages[-max_messages:]
            logger.info(f"对话 {chat_id} 消息数量超过限制，已截断")
    
    def set_role(self, user_id: int, chat_id: int, role_name: str) -> bool:
        """设置对话角色"""
        try:
            # 从配置文件获取角色
            role_config = config.get_role(role_name)
            if not role_config:
                logger.warning(f"角色 {role_name} 不存在")
                return False
            
            # 创建角色对象
            role = Role(
                name=role_config.name,
                english_name=role_config.english_name,
                description=role_config.description,
                personality=role_config.personality,
                greeting=role_config.greeting,
                farewell=role_config.farewell
            )
            
            # 更新对话角色
            conversation = self.get_conversation(user_id, chat_id)
            conversation.role = role
            conversation.last_activity = time.time()
            
            # 更新角色使用统计
            if self.database_service:
                try:
                    import asyncio
                    asyncio.create_task(self.database_service.update_role_stats(role_name))
                except Exception as e:
                    logger.warning(f"更新角色统计失败: {e}")
            
            logger.info(f"用户 {user_id} 在聊天 {chat_id} 中设置角色为: {role_name}")
            return True
            
        except Exception as e:
            logger.error(f"设置角色失败: {e}")
            return False
    
    def get_role(self, user_id: int, chat_id: int) -> Optional[Role]:
        """获取当前对话角色"""
        conversation = self.get_conversation(user_id, chat_id)
        return conversation.role
    
    def list_roles(self) -> List[str]:
        """列出所有可用角色"""
        return list(config.roles.keys())
    
    def create_role(self, user_id: int, chat_id: int, role_data: Dict[str, Any]) -> bool:
        """创建自定义角色"""
        try:
            # 创建新角色
            role = Role(
                name=role_data.get("name", "自定义角色"),
                english_name=role_data.get("english_name", "Custom Role"),
                description=role_data.get("description", "用户自定义角色"),
                personality=role_data.get("personality", "你是一个自定义角色。"),
                greeting=role_data.get("greeting", "你好！"),
                farewell=role_data.get("farewell", "再见！")
            )
            
            # 添加到角色列表
            role_name = f"custom_{user_id}_{int(time.time())}"
            self.roles[role_name] = role
            
            # 设置为当前对话角色
            conversation = self.get_conversation(user_id, chat_id)
            conversation.role = role
            conversation.last_activity = time.time()
            
            logger.info(f"用户 {user_id} 创建自定义角色: {role.name}")
            return True
            
        except Exception as e:
            logger.error(f"创建角色失败: {e}")
            return False
    
    def reset_conversation(self, user_id: int, chat_id: int) -> bool:
        """重置对话"""
        try:
            key = f"{user_id}_{chat_id}"
            if key in self.conversations:
                # 重置消息，但保留角色
                conversation = self.conversations[key]
                conversation.messages = []
                conversation.last_activity = time.time()
                logger.info(f"重置用户 {user_id} 在聊天 {chat_id} 的对话")
                return True
            return False
        except Exception as e:
            logger.error(f"重置对话失败: {e}")
            return False
    
    def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        """获取用户统计信息"""
        try:
            user_conversations = [
                conv for conv in self.conversations.values() 
                if conv.user_id == user_id
            ]
            
            total_messages = sum(len(conv.messages) for conv in user_conversations)
            active_conversations = len(user_conversations)
            
            return {
                "user_id": user_id,
                "total_messages": total_messages,
                "active_conversations": active_conversations,
                "current_role": user_conversations[0].role.name if user_conversations else "无"
            }
        except Exception as e:
            logger.error(f"获取用户统计失败: {e}")
            return {}
    
    def cleanup_expired_conversations(self, max_age: int = 3600) -> int:
        """清理过期对话"""
        try:
            current_time = time.time()
            expired_keys = []
            
            for key, conversation in self.conversations.items():
                if current_time - conversation.last_activity > max_age:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.conversations[key]
            
            if expired_keys:
                logger.info(f"清理了 {len(expired_keys)} 个过期对话")
            
            return len(expired_keys)
        except Exception as e:
            logger.error(f"清理过期对话失败: {e}")
            return 0
