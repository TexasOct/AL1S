"""
数据模型模块
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import time


class Message(BaseModel):
    """聊天消息模型"""
    role: str  # "user" 或 "assistant"
    content: str
    timestamp: float  # Unix时间戳


class Role(BaseModel):
    """角色模型"""
    name: str
    english_name: Optional[str] = None
    description: str
    personality: str  # 角色性格设定
    greeting: str     # 角色问候语
    farewell: str     # 角色告别语
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class Conversation(BaseModel):
    """对话模型"""
    user_id: int
    chat_id: int
    role: Optional[Role] = None
    messages: List[Message] = Field(default_factory=list)
    created_at: float = Field(default_factory=lambda: time.time())
    last_activity: float = Field(default_factory=lambda: time.time())


class ChatResponse(BaseModel):
    """聊天响应模型"""
    text: str
    role: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ImageSearchResult(BaseModel):
    """图片搜索结果模型"""
    source: str
    url: str
    title: Optional[str] = None
    similarity: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class Command(BaseModel):
    """命令模型"""
    name: str
    description: str
    usage: str
    aliases: List[str] = Field(default_factory=list)
    requires_args: bool = False
    admin_only: bool = False


class User(BaseModel):
    """用户模型"""
    id: int
    username: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    is_bot: bool = False
    language_code: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    last_seen: datetime = Field(default_factory=datetime.now)
    preferences: Dict[str, Any] = Field(default_factory=dict)
