"""
配置管理模块
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import yaml


class OpenAIConfig(BaseModel):
    """OpenAI配置"""
    api_key: str = Field(..., description="OpenAI API密钥")
    base_url: str = Field("https://api.openai.com/v1", description="OpenAI API基础URL")
    model: str = Field("gpt-4o-mini", description="使用的模型名称")
    max_tokens: int = Field(2000, description="最大生成token数")
    temperature: float = Field(0.7, description="生成温度")
    timeout: int = Field(60, description="API超时时间（秒）")


class TelegramConfig(BaseModel):
    """Telegram配置"""
    bot_token: str = Field(..., description="Telegram机器人token")
    webhook_url: Optional[str] = Field(None, description="Webhook URL（可选）")
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


class AppConfig(BaseSettings):
    """应用配置"""
    openai: OpenAIConfig
    telegram: TelegramConfig
    ascii2d: Ascii2DConfig
    
    # 角色配置
    roles: Dict[str, RoleConfig] = Field(default_factory=dict, description="角色配置字典")
    default_role: str = Field("天童爱丽丝", description="默认角色名称")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 加载角色配置文件
        self._load_roles_config()
    
    def _load_roles_config(self):
        """加载角色配置文件"""
        try:
            roles_file = Path(__file__).parent / "config" / "roles.yaml"
            if roles_file.exists():
                with open(roles_file, 'r', encoding='utf-8') as f:
                    roles_data = yaml.safe_load(f)
                
                # 设置默认角色
                if 'default_role' in roles_data:
                    self.default_role = roles_data['default_role']
                
                # 加载角色配置
                if 'roles' in roles_data:
                    for role_name, role_data in roles_data['roles'].items():
                        self.roles[role_name] = RoleConfig(**role_data)
                
                print(f"成功加载 {len(self.roles)} 个角色配置")
            else:
                print(f"角色配置文件不存在: {roles_file}")
        except Exception as e:
            print(f"加载角色配置文件失败: {e}")
    
    def get_role(self, role_name: str) -> Optional[RoleConfig]:
        """获取指定角色配置"""
        return self.roles.get(role_name)
    
    def get_default_role(self) -> Optional[RoleConfig]:
        """获取默认角色配置"""
        return self.roles.get(self.default_role)
    
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
            timeout=int(os.getenv("OPENAI_TIMEOUT", "60"))
        ),
        telegram=TelegramConfig(
            bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            webhook_url=os.getenv("TELEGRAM_WEBHOOK_URL"),
            webhook_port=int(os.getenv("TELEGRAM_WEBHOOK_PORT", "8443"))
        ),
        ascii2d=Ascii2DConfig(
            base_url=os.getenv("ASCII2D_BASE_URL", "https://ascii2d.net"),
            bovw=os.getenv("ASCII2D_BOVW", "false").lower() == "true"
        )
    )


# 全局配置实例
config = AppConfig()
