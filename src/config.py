"""
配置管理模块
"""

import os
import tomllib
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class OpenAIConfig(BaseModel):
    """OpenAI配置"""

    api_key: str = Field("", description="OpenAI API密钥")
    base_url: str = Field("https://api.openai.com/v1", description="OpenAI API基础URL")
    model: str = Field("gpt-4o-mini", description="使用的模型名称")
    max_tokens: int = Field(2000, description="最大生成token数")
    temperature: float = Field(0.7, description="生成温度")
    timeout: int = Field(60, description="API超时时间（秒）")


class TelegramConfig(BaseModel):
    """Telegram配置"""

    bot_token: str = Field("", description="Telegram机器人token")
    webhook_url: Optional[str] = Field("", description="Webhook URL（可选）")
    webhook_port: int = Field(8443, description="Webhook端口")


class Ascii2DConfig(BaseModel):
    """Ascii2D配置"""

    base_url: str = Field("https://ascii2d.net", description="Ascii2D基础URL")
    bovw: bool = Field(False, description="是否使用特征搜索")


class RoleConfig(BaseModel):
    """角色配置"""

    name: str = Field(..., description="角色名称")
    english_name: str = Field(..., description="角色英文名")
    description: str = Field(..., description="角色描述")
    personality: str = Field(..., description="角色性格设定")
    greeting: str = Field(..., description="角色问候语")
    farewell: str = Field(..., description="角色告别语")


class MCPServerConfig(BaseModel):
    """MCP服务器配置"""

    name: str = Field(..., description="服务器名称")
    command: str = Field(..., description="启动命令")
    args: list[str] = Field(default_factory=list, description="命令参数")
    env: dict[str, str] = Field(default_factory=dict, description="环境变量")
    enabled: bool = Field(True, description="是否启用")


class MCPConfig(BaseModel):
    """MCP配置"""

    enabled: bool = Field(True, description="是否启用MCP功能")
    servers: list[MCPServerConfig] = Field(
        default_factory=list, description="MCP服务器列表"
    )


class RAGConfig(BaseModel):
    """RAG配置"""

    enabled: bool = Field(True, description="是否启用RAG功能")
    vector_store_path: str = Field("data/vector_store", description="向量存储路径")
    embedding_model: str = Field("tfidf", description="嵌入模型类型")
    max_knowledge_entries: int = Field(10000, description="最大知识条目数")
    similarity_threshold: float = Field(0.3, description="相似度阈值")
    top_k_retrieval: int = Field(5, description="检索返回的最大条目数")
    auto_learning: bool = Field(True, description="是否自动从对话中学习")
    learning_trigger_messages: int = Field(3, description="学习触发的最小消息数")
    importance_threshold: float = Field(0.1, description="知识重要性阈值")
    use_llm_extraction: bool = Field(True, description="是否使用LLM进行高级知识提取")


class AgentConfig(BaseModel):
    """Agent 配置"""

    type: str = Field("unified", description="Agent 类型: unified | langchain")

    # 通用配置
    vector_store: str = Field("faiss", description="向量存储类型: memory | faiss")
    embedding_model: str = Field(
        "tfidf",
        description="嵌入模型: tfidf | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    vector_store_path: str = Field("data/vector_store", description="向量存储路径")

    # 学习配置
    auto_learning: bool = Field(True, description="是否启用自动学习")
    learning_threshold: float = Field(0.8, description="学习阈值")


class LangChainConfig(BaseModel):
    """LangChain 特定配置"""
    # enabled 属性由 agent.type 自动控制，不再作为配置项
    vector_store: str = Field("faiss", description="向量存储类型: memory | faiss")
    embedding: str = Field(
        "huggingface_bge_m3", description="嵌入模型提供者：openai | huggingface_bge_m3"
    )
    embedding_model_name: str = Field("BAAI/bge-m3", description="HF嵌入模型名称")
    embedding_device: str = Field("cpu", description="嵌入推理设备：cpu | cuda")
    model_cache_dir: str = Field("data/models", description="模型缓存目录")
    download_timeout: int = Field(300, description="模型下载超时时间（秒）")
    download_retries: int = Field(5, description="下载重试次数")
    retriever_k: int = Field(5, description="检索topK")
    chunk_size: int = Field(1000, description="文本分块大小")
    chunk_overlap: int = Field(200, description="文本分块重叠")
    
    # 动态属性，由配置验证时设置
    enabled: bool = Field(default=False, description="是否启用（由 agent.type 自动控制）")


class AppConfig:
    """应用配置"""

    def __init__(self, **kwargs):
        # 初始化属性
        self.roles = {}
        self.default_role = "天童爱丽丝"

        # 先加载统一配置文件
        config_data = self._load_unified_config()

        # 从TOML配置中提取各个配置部分
        openai_config = config_data.get("openai", {}) if config_data else {}
        telegram_config = config_data.get("telegram", {}) if config_data else {}
        ascii2d_config = config_data.get("ascii2d", {}) if config_data else {}
        mcp_config = config_data.get("mcp", {}) if config_data else {}
        rag_config = config_data.get("rag", {}) if config_data else {}
        agent_config = config_data.get("agent", {}) if config_data else {}
        lc_config = config_data.get("langchain", {}) if config_data else {}

        # 初始化各个配置对象，使用默认值填充缺失的配置
        self.openai = OpenAIConfig(**openai_config)
        self.telegram = TelegramConfig(**telegram_config)
        self.ascii2d = Ascii2DConfig(**ascii2d_config)
        self.mcp = MCPConfig(**mcp_config)
        self.rag = RAGConfig(**rag_config)
        self.agent = AgentConfig(**agent_config)
        self.langchain = LangChainConfig(**lc_config)

        # 验证 Agent 配置互斥性
        self._validate_agent_config()

        # 设置角色配置
        if config_data and "default_role" in config_data:
            self.default_role = config_data["default_role"]

        if config_data and "roles" in config_data:
            for role_data in config_data["roles"]:
                role_name = role_data["name"]
                self.roles[role_name] = RoleConfig(**role_data)

        # 验证必需配置
        self._validate_config()

    def _load_unified_config(self):
        """加载统一配置文件"""
        try:
            config_file = Path(__file__).parent.parent / "config.toml"
            if config_file.exists():
                with open(config_file, "rb") as f:
                    config_data = tomllib.load(f)

                print(f"成功加载配置文件: {config_file}")
                return config_data
            else:
                print(f"配置文件不存在: {config_file}")
                print(
                    "请复制 config/config.toml.example 为 config/config.toml 并填写配置信息"
                )
                return {}
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            print("请检查配置文件格式是否正确")
            return {}

    def _validate_config(self):
        """验证必需配置"""
        if not self.openai.api_key:
            print("⚠️  警告: OpenAI API密钥未设置")
        if not self.telegram.bot_token:
            print("⚠️  警告: Telegram Bot Token未设置")
        if not self.roles:
            print("⚠️  警告: 未加载任何角色配置")

    def get_role(self, role_name: str) -> Optional[RoleConfig]:
        """获取指定角色配置"""
        return self.roles.get(role_name)

    def get_default_role(self) -> Optional[RoleConfig]:
        """获取默认角色配置"""
        return self.roles.get(self.default_role)

    def _validate_agent_config(self):
        """验证 Agent 配置"""
        from loguru import logger

        # 根据 agent.type 自动设置相关配置
        if self.agent.type == "langchain":
            # 使用 LangChain Agent 时自动启用 langchain 配置
            self.langchain.enabled = True
            logger.info("使用 LangChain Agent")
        elif self.agent.type == "unified":
            # 使用统一 Agent 时禁用 langchain 配置
            self.langchain.enabled = False
            logger.info("使用统一 Agent")
        else:
            # 未知类型回退到统一 Agent
            logger.warning(f"未知的 Agent 类型: {self.agent.type}，回退到统一 Agent")
            self.agent.type = "unified"
            self.langchain.enabled = False

    def list_roles(self) -> list:
        """列出所有可用角色"""
        return list(self.roles.keys())


# 加载配置
def load_config() -> AppConfig:
    """加载应用配置"""
    return AppConfig(
        openai=OpenAIConfig(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "2000")),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
            timeout=int(os.getenv("OPENAI_TIMEOUT", "60")),
        ),
        telegram=TelegramConfig(
            bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            webhook_url=os.getenv("TELEGRAM_WEBHOOK_URL"),
            webhook_port=int(os.getenv("TELEGRAM_WEBHOOK_PORT", "8443")),
        ),
        ascii2d=Ascii2DConfig(
            base_url=os.getenv("ASCII2D_BASE_URL", "https://ascii2d.net"),
            bovw=os.getenv("ASCII2D_BOVW", "false").lower() == "true",
        ),
    )


# 全局配置实例
config = AppConfig()
