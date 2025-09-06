# AL1S-Bot 🤖

基于《蔚蓝档案》天童爱丽丝角色的智能 Telegram 机器人，集成 AI 对话、知识学习、工具调用等功能。

## ✨ 核心特性

- **🎭 角色扮演**: 天童爱丽丝等多种预设角色，支持角色切换
- **🧠 智能学习**: 自动从对话中学习并记忆用户信息
- **🔧 工具集成**: 支持 MCP 协议，可调用文件系统、GitHub、搜索等工具
- **🔍 图片搜索**: 基于 Ascii2D 的图片反向搜索功能
- **💾 持久存储**: SQLite 数据库存储对话历史和知识库
- **🌐 多模型支持**: 兼容 OpenAI、月之暗面、DeepSeek 等 API

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd AL1S-Bot

# 安装依赖（推荐使用 uv）
uv sync

# 或使用 pip
pip install -r requirements.txt
```

### 2. 配置设置

```bash
# 复制配置模板
cp config.example.toml config.toml

# 编辑配置文件，填入必要信息
nano config.toml
```

**必需配置**：
```toml
[openai]
api_key = "your-api-key-here"
base_url = "https://api.openai.com/v1"  # 或其他兼容API

[telegram]
bot_token = "your-telegram-bot-token"
```

### 3. 初始化数据库

```bash
# 自动创建数据库（首次运行时）
mkdir -p data
sqlite3 data/bot.db < data/init_db.sql
```

### 4. 启动机器人

```bash
# 使用 uv 运行
uv run python main.py

# 或直接运行
python main.py
```

## 📁 项目架构

```
AL1S-Bot/
├── src/
│   ├── agents/              # Agent 实现
│   │   ├── unified_agent_service.py    # 统一 Agent
│   │   └── langchain_agent_service.py  # LangChain Agent
│   ├── infra/               # 基础设施层
│   │   ├── database.py      # 数据库服务
│   │   ├── vector.py        # 向量存储服务
│   │   └── mcp.py          # MCP 工具集成
│   ├── services/            # 业务服务层
│   │   ├── conversation_service.py     # 对话管理
│   │   ├── learning_service.py         # 知识学习
│   │   └── ascii2d_service.py          # 图片搜索
│   ├── handlers/            # 消息处理器
│   │   ├── chat_handler.py             # 聊天处理
│   │   ├── command_handler.py          # 命令处理
│   │   └── image_handler.py            # 图片处理
│   ├── models.py            # 数据模型
│   ├── config.py            # 配置管理
│   └── bot.py              # 机器人主类
├── data/                    # 数据目录
├── config.example.toml      # 配置模板
└── main.py                 # 程序入口
```

## 🎮 使用指南

### 基础命令

- `/start` - 开始使用
- `/help` - 显示帮助
- `/ping` - 测试连接

### 角色管理

- `/role` - 查看当前角色
- `/role <角色名>` - 切换角色
- `/roles` - 显示所有角色

### 对话管理

- `/reset` - 重置对话
- `/stats` - 对话统计

### 知识管理

- `/knowledge search <关键词>` - 搜索知识库
- `/knowledge stats` - 知识库统计
- `/rebuild_knowledge` - 重建知识索引

## ⚙️ 配置详解

### Agent 配置

```toml
[agent]
# Agent 类型选择（这是唯一的控制开关）
type = "unified"  # 或 "langchain"

# 向量存储配置
vector_store = "faiss"
vector_store_path = "data/vector_store"

# 嵌入模型选择
embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# 自动学习
auto_learning = true
learning_threshold = 0.8
```

**重要说明**：
- `agent.type` 是唯一的 Agent 控制开关
- 当设置为 `"langchain"` 时，自动启用 LangChain 相关功能
- 当设置为 `"unified"` 时，使用统一 Agent，自动禁用 LangChain
- 无需手动设置 `langchain.enabled`，系统会自动处理

### MCP 工具配置

```toml
[mcp]
enabled = true

# 文件系统工具
[[mcp.servers]]
name = "filesystem"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "."]
enabled = true

# GitHub 工具
[[mcp.servers]]
name = "github"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-github"]
enabled = false
env = { GITHUB_PERSONAL_ACCESS_TOKEN = "your-token" }
```

### 角色自定义

```toml
[[roles]]
name = "自定义角色"
english_name = "Custom Role"
description = "角色描述"
personality = """
角色的详细设定和性格描述...
"""
greeting = "角色问候语"
farewell = "角色告别语"
```

## 🔧 高级功能

### 1. 智能学习系统

机器人会自动从对话中学习：
- 个人信息（生日、喜好等）
- 问答对话
- 重要事实
- 用户习惯

### 2. 工具调用能力

通过 MCP 协议支持：
- 文件系统操作
- GitHub 仓库查询
- 网络搜索
- 数据库查询
- 自定义工具扩展

### 3. 多模态处理

- 文本对话
- 图片分析和搜索
- 文件处理

## 🚨 故障排除

### 常见问题

1. **启动失败**
   ```bash
   # 检查配置文件
   python -c "import tomllib; print('配置OK' if tomllib.load(open('config.toml', 'rb')) else '配置错误')"
   ```

2. **API 连接问题**
   ```bash
   # 测试 API 连接
   curl -H "Authorization: Bearer YOUR_API_KEY" YOUR_BASE_URL/models
   ```

3. **数据库问题**
   ```bash
   # 重新初始化数据库
   rm data/bot.db
   sqlite3 data/bot.db < data/init_db.sql
   ```

4. **向量存储问题**
   ```bash
   # 清理并重建向量存储
   rm -rf data/vector_store/*
   # 重启机器人会自动重建
   ```

### 性能优化

- **内存使用**: 语义模型需要约 800MB 内存
- **存储空间**: 建议预留 1GB 存储空间
- **网络要求**: 首次运行需下载模型（约 400MB）

## 🔒 安全注意事项

1. **API 密钥**: 妥善保管 API 密钥，不要提交到版本控制
2. **访问权限**: 合理配置 MCP 工具的访问权限
3. **数据隐私**: 定期清理敏感对话记录
4. **网络安全**: 在生产环境中使用 HTTPS 和适当的防火墙设置

## 📊 监控和维护

### 日志管理

```bash
# 查看日志
tail -f logs/bot.log

# 按时间查看
ls logs/bot.*.log
```

### 数据库维护

```bash
# 数据库统计
sqlite3 data/bot.db "
SELECT 
  'messages' as table_name, COUNT(*) as count FROM messages
UNION ALL
SELECT 
  'knowledge_entries' as table_name, COUNT(*) as count FROM knowledge_entries;
"

# 清理旧数据（保留最近30天）
sqlite3 data/bot.db "
DELETE FROM messages 
WHERE created_at < datetime('now', '-30 days');
"
```

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- [OpenAI](https://openai.com/) - AI 模型支持
- [Sentence Transformers](https://www.sbert.net/) - 语义嵌入模型
- [FAISS](https://faiss.ai/) - 向量相似性搜索
- [Model Context Protocol](https://modelcontextprotocol.io/) - 工具集成协议
- [python-telegram-bot](https://python-telegram-bot.org/) - Telegram Bot 框架

---

**邦邦卡邦！** 如有问题，请提交 Issue 或联系维护者。