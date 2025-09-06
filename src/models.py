"""
数据模型模块
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


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
    greeting: str  # 角色问候语
    farewell: str  # 角色告别语
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


class KnowledgeEntry:
    """知识条目类"""

    def __init__(
        self,
        id: int = None,
        user_id: int = None,
        conversation_id: int = None,
        title: str = "",
        content: str = "",
        summary: str = "",
        keywords: str = "",
        category: str = "general",
        importance_score: float = 0.0,
        embedding_id: str = None,
        source_message_id: int = None,
        created_at=None,
    ):
        self.id = id
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.title = title
        self.content = content
        self.summary = summary
        self.keywords = keywords
        self.category = category
        self.importance_score = importance_score
        self.embedding_id = embedding_id
        self.source_message_id = source_message_id
        self.created_at = created_at or datetime.now()

    def to_dict(self):
        """转换为字典"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "title": self.title,
            "content": self.content,
            "summary": self.summary,
            "keywords": self.keywords,
            "category": self.category,
            "importance_score": self.importance_score,
            "embedding_id": self.embedding_id,
            "source_message_id": self.source_message_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
