# AL1S-Bot 🤖

一个功能完整的Telegram聊天机器人，支持OpenAI聊天、角色定制和图片搜索功能。

## ✨ 主要功能

- **💬 智能对话**: 基于OpenAI的AI助手，支持多种模型
- **🎭 角色定制**: 预设多种角色，支持自定义角色创建
- **🔍 图片搜索**: 通过PicImageSearch库支持Ascii2D等搜索引擎
- **⚙️ 命令系统**: 丰富的管理命令和配置选项
- **🔄 对话管理**: 智能对话上下文管理和角色切换

## 🚀 快速开始

### 1. 环境要求

- Python 3.11+
- uv 包管理器（推荐）或 pip
- Telegram Bot Token
- OpenAI API Key

### 2. 安装依赖

```bash
# 使用uv安装依赖（推荐）
uv sync

# 或者使用pip
pip install -r requirements.txt
```

### 3. 配置环境变量

复制 `env.example` 为 `.env` 并填入配置：

```bash
cp env.example .env
```

编辑 `.env` 文件，填入必要的配置：

```env
# OpenAI配置
OPENAI__API_KEY=your_openai_api_key_here
OPENAI__MODEL=gpt-3.5-turbo

# Telegram配置
TELEGRAM__BOT_TOKEN=your_telegram_bot_token_here
```

### 4. 运行机器人

```bash
# 使用启动脚本（推荐）
chmod +x start.sh
./start.sh

# 或者直接运行
python main.py
```

## 📁 项目结构

```
AL1S-Bot/
├── src/                    # 源代码目录
│   ├── config.py          # 配置管理
│   ├── models.py          # 数据模型
│   ├── bot.py             # 机器人主类
│   ├── services/          # 服务层
│   │   ├── openai_service.py      # OpenAI服务
│   │   ├── ascii2d_service.py     # Ascii2D服务（基于PicImageSearch）
│   │   └── conversation_service.py # 对话管理服务
│   └── handlers/          # 处理器层
│       ├── base_handler.py        # 基础处理器
│       ├── chat_handler.py        # 聊天处理器
│       ├── command_handler.py     # 命令处理器
│       └── image_handler.py       # 图片处理器
├── main.py                # 主程序入口
├── start.sh               # 启动脚本
├── pyproject.toml         # 项目配置
├── requirements.txt       # 依赖列表（备选）
├── env.example            # 环境变量示例
└── README.md              # 项目说明
```

## 🎮 使用说明

### 基础命令

- `/start` - 开始使用机器人
- `/help` - 显示帮助信息
- `/ping` - 测试机器人响应

### 角色管理

- `/role` - 查看当前角色
- `/role 角色名` - 切换到指定角色
- `/roles` - 显示所有可用角色
- `/create_role 名称 描述` - 创建自定义角色

### 对话管理

- `/reset` - 重置当前对话
- `/stats` - 显示对话统计信息

### 图片搜索

- `/search 图片URL` - 搜索指定URL的图片
- `/search_engines` - 显示可用的搜索引擎
- `/test_search` - 测试图片搜索服务
- 发送图片文件 - 自动分析并搜索相似图片

### 预设角色

- **default**: 默认AI助手
- **teacher**: 耐心的老师
- **friend**: 贴心的朋友
- **expert**: 专业专家

## 🔍 图片搜索功能

### 支持的搜索引擎

- **Ascii2D**: 基于[PicImageSearch](https://pic-image-search.kituin.fun/)库
  - 支持二次元图片搜索
  - 支持URL和文件上传搜索
  - 提供相似度评分
  - 自动解析搜索结果

### 搜索方式

1. **文件上传**: 直接发送图片文件到机器人
2. **URL搜索**: 使用 `/search 图片URL` 命令
3. **多引擎搜索**: 支持未来扩展多个搜索引擎

### 搜索结果

- 图片分析结果（基于OpenAI Vision）
- 相似图片列表
- 来源信息和相似度评分
- 直接访问链接

## ⚙️ 配置选项

### OpenAI配置

- `OPENAI__API_KEY`: OpenAI API密钥
- `OPENAI__BASE_URL`: API基础URL（支持代理）
- `OPENAI__MODEL`: 使用的模型名称
- `OPENAI__MAX_TOKENS`: 最大token数量
- `OPENAI__TEMPERATURE`: 温度参数

### Telegram配置

- `TELEGRAM__BOT_TOKEN`: 机器人Token
- `TELEGRAM__WEBHOOK_URL`: Webhook URL（可选）
- `TELEGRAM__MAX_CONVERSATION_AGE`: 对话最大保存时间

### 角色配置

- `ROLE__DEFAULT_ROLE`: 默认角色设定
- `ROLE__SYSTEM_PROMPT`: 系统提示词
- `ROLE__MAX_CONTEXT_LENGTH`: 最大上下文长度

### Ascii2D配置

- `ASCII2D__BASE_URL`: Ascii2D基础URL
- `ASCII2D__TIMEOUT`: 请求超时时间
- `ASCII2D__MAX_RESULTS`: 最大搜索结果数量

## 🔧 开发说明

### 添加新处理器

1. 继承 `BaseHandler` 类
2. 实现 `handle` 方法
3. 在 `bot.py` 中注册处理器

### 添加新服务

1. 在 `services/` 目录下创建服务类
2. 实现必要的接口方法
3. 在 `bot.py` 中初始化服务

### 添加新命令

1. 在 `CommandHandler` 中添加命令定义
2. 实现对应的处理方法
3. 在 `bot.py` 中注册命令处理器

### 扩展图片搜索

1. 在 `Ascii2DService` 中添加新的搜索引擎
2. 实现对应的搜索方法
3. 更新配置和命令处理

## 🐛 故障排除

### 常见问题

1. **配置错误**: 检查 `.env` 文件中的配置是否正确
2. **API密钥无效**: 确认OpenAI API密钥和Telegram Bot Token有效
3. **网络问题**: 检查网络连接和代理设置
4. **权限问题**: 确认机器人有足够的权限
5. **图片搜索失败**: 检查PicImageSearch库是否正确安装

### 日志查看

日志文件保存在 `logs/` 目录下，可以通过查看日志来诊断问题。

### 依赖问题

如果遇到依赖问题，可以尝试：

```bash
# 清理并重新安装
uv clean
uv sync

# 或者使用pip
pip uninstall -r requirements.txt
pip install -r requirements.txt
```

## 📝 许可证

本项目采用 MIT 许可证。

## 🤝 贡献

欢迎提交Issue和Pull Request！

### 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 📞 支持

如有问题，请通过以下方式联系：

- 提交GitHub Issue
- 发送邮件至项目维护者

## 🙏 致谢

- [PicImageSearch](https://pic-image-search.kituin.fun/) - 提供强大的图片搜索功能
- [python-telegram-bot](https://python-telegram-bot.org/) - Telegram Bot API的Python实现
- [OpenAI](https://openai.com/) - 提供AI对话能力

---

**注意**: 请确保遵守OpenAI和Telegram的使用条款，合理使用API资源。