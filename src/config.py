"""
配置管理模块
"""
import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
import tomllib


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


class AppConfig:
    """应用配置"""

    
    def __init__(self, **kwargs):
        # 初始化属性
        self.roles = {}
        self.default_role = "天童爱丽丝"
        
        # 先加载统一配置文件
        config_data = self._load_unified_config()
        
        # 从TOML配置中提取各个配置部分
        openai_config = config_data.get('openai', {}) if config_data else {}
        telegram_config = config_data.get('telegram', {}) if config_data else {}
        ascii2d_config = config_data.get('ascii2d', {}) if config_data else {}
        
        # 初始化各个配置对象，使用默认值填充缺失的配置
        self.openai = OpenAIConfig(**openai_config)
        self.telegram = TelegramConfig(**telegram_config)
        self.ascii2d = Ascii2DConfig(**ascii2d_config)
        
        # 设置角色配置
        if config_data and 'default_role' in config_data:
            self.default_role = config_data['default_role']
        
        if config_data and 'roles' in config_data:
            for role_data in config_data['roles']:
                role_name = role_data['name']
                self.roles[role_name] = RoleConfig(**role_data)
        
        # 验证必需配置
        self._validate_config()
    
    def _load_unified_config(self):
        """加载统一配置文件"""
        try:
            config_file = Path(__file__).parent.parent / "config.toml"
            if config_file.exists():
                with open(config_file, 'rb') as f:
                    config_data = tomllib.load(f)
                
                print(f"成功加载配置文件: {config_file}")
                return config_data
            else:
                print(f"配置文件不存在: {config_file}")
                print("请复制 config/config.toml.example 为 config/config.toml 并填写配置信息")
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
