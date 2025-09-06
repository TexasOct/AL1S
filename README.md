# AL1S-Bot 🤖

一个功能完整的Telegram聊天机器人，支持OpenAI聊天、RAG知识增强、角色定制和图片搜索功能。

## ✨ 主要功能

- **💬 智能对话**: 基于OpenAI的AI助手，支持多种模型
- **🧠 RAG知识增强**: 智能学习和记忆用户信息，支持中文语义检索
- **🎭 角色定制**: 预设多种角色，支持自定义角色创建
- **🔍 图片搜索**: 通过PicImageSearch库支持Ascii2D等搜索引擎
- **🛠️ MCP工具集成**: 支持Model Context Protocol工具调用
- **⚙️ 命令系统**: 丰富的管理命令和配置选项
- **🔄 对话管理**: 智能对话上下文管理和角色切换
- **📊 数据持久化**: SQLite数据库存储，支持对话历史和知识库

## 🚀 快速开始

### 1. 环境要求

- Python 3.11+
- uv 包管理器（推荐）或 pip
- Telegram Bot Token
- OpenAI API Key
- 至少 2GB 内存（用于加载语义模型）
- 网络连接（首次运行会下载语义模型）

### 2. 安装依赖

```bash
# 使用uv安装依赖（推荐）
uv sync

# 或者使用pip
pip install -r requirements.txt
```

### 3. 配置设置

复制 `config.example.toml` 为 `config.toml` 并填入配置：

```bash
cp config.example.toml config.toml
```

编辑 `config.toml` 文件，填入必要的配置：

```toml
# OpenAI配置
[openai]
api_key = "your_openai_api_key_here"
model = "gpt-3.5-turbo"
base_url = "https://api.openai.com/v1"

# Telegram配置  
[telegram]
bot_token = "your_telegram_bot_token_here"

# RAG配置
[rag]
enabled = true
embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

### 4. 初始化数据库

```bash
# 初始化SQLite数据库
sqlite3 data/bot.db < data/init_db.sql
```

### 5. 运行机器人

```bash
# 使用uv运行（推荐）
uv run python main.py

# 或者直接运行
python main.py
```

**首次运行说明**: 如果启用了RAG功能，首次运行时会自动下载中文语义模型（约400MB），请保持网络连接。

## 📁 项目结构

```
AL1S-Bot/
├── src/                    # 源代码目录
│   ├── config.py          # 配置管理
│   ├── models.py          # 数据模型
│   ├── bot.py             # 机器人主类
│   ├── services/          # 服务层
│   │   ├── openai_service.py      # OpenAI服务
│   │   ├── rag_service.py         # RAG知识增强服务
│   │   ├── knowledge_extractor.py # 知识提取器
│   │   ├── ascii2d_service.py     # Ascii2D服务
│   │   ├── mcp_service.py         # MCP工具服务
│   │   ├── database_service.py    # 数据库服务
│   │   └── conversation_service.py # 对话管理服务
│   ├── handlers/          # 处理器层
│   │   ├── base_handler.py        # 基础处理器
│   │   ├── chat_handler.py        # 聊天处理器
│   │   ├── command_handler.py     # 命令处理器
│   │   └── image_handler.py       # 图片处理器
│   └── utils/             # 工具层
│       └── database_logger.py     # 数据库日志
├── data/                  # 数据目录
│   └─── init_db.sql       # 数据库初始化脚本
├── main.py               # 主程序入口
├── pyproject.toml        # 项目配置
├── config.example.toml   # 配置文件示例
├── docker-compose.yml    # Docker配置
├── Dockerfile           # Docker镜像
└── README.md            # 项目说明
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

### RAG知识管理

- `/rag_stats` - 显示RAG统计信息
- `/knowledge search 关键词` - 搜索知识库
- `/learn 内容` - 手动添加知识
- `/forget 知识ID` - 删除指定知识
- `/rebuild_index` - 重建向量索引

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
- **alice**: 爱丽丝角色（活泼可爱的AI助手）

## 🧠 RAG知识增强功能

### 核心特性

- **智能学习**: 自动从对话中提取和学习有价值的信息
- **语义检索**: 基于中文语义模型的智能检索
- **个人记忆**: 记住用户的个人信息（生日、喜好等）
- **持久存储**: 知识永久保存，重启后仍然有效
- **多模态支持**: 支持TF-IDF和Sentence Transformers两种嵌入模型

### 支持的嵌入模型

1. **TF-IDF模型** (轻量快速)
   ```toml
   embedding_model = "tfidf"
   ```

2. **多语言语义模型** (推荐中文)
   ```toml
   embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
   ```

3. **轻量多语言模型**
   ```toml
   embedding_model = "sentence-transformers/distiluse-base-multilingual-cased"
   ```

### 知识类型

- **个人信息**: 生日、姓名、职业、爱好等
- **问答对**: 用户问题和助手回答
- **定义解释**: 概念定义和详细说明
- **列表步骤**: 操作步骤和流程说明
- **事实信息**: 重要的事实性内容

### 自动学习触发条件

- 对话消息数达到设定阈值（默认2条）
- 内容重要性评分超过阈值（默认0.05）
- 检测到个人信息分享
- 包含问答对话模式

### 语义检索示例

输入"生日"可以检索到：
- "我的生日是9月4日"
- "生日是什么时候"
- "老师的生日"
- "出生日期"
- "birthday"等相关内容

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

所有配置都在 `config.toml` 文件中设置。

### OpenAI配置

```toml
[openai]
api_key = "your_api_key"           # OpenAI API密钥
base_url = "https://api.openai.com/v1"  # API基础URL
model = "gpt-3.5-turbo"            # 使用的模型名称
max_tokens = 2000                  # 最大token数量
temperature = 0.7                  # 温度参数 (0.0-2.0)
timeout = 60                       # 请求超时时间（秒）
```

### Telegram配置

```toml
[telegram]
bot_token = "your_bot_token"       # 机器人Token
max_conversation_age = 86400       # 对话最大保存时间（秒）
```

### RAG配置

```toml
[rag]
enabled = true                     # 是否启用RAG功能
vector_store_path = "data/vector_store"  # 向量存储路径
embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
max_knowledge_entries = 10000      # 最大知识条目数
similarity_threshold = 0.15        # 相似度阈值
top_k_retrieval = 8               # 检索返回的最大条目数
auto_learning = true              # 是否自动从对话中学习
learning_trigger_messages = 2     # 学习触发的最小消息数
importance_threshold = 0.05       # 知识重要性阈值
use_llm_extraction = true         # 是否使用LLM进行高级知识提取
```

### 角色配置

```toml
[role]
default_role = "alice"            # 默认角色
system_prompt = "你是一个智能助手"  # 系统提示词
max_context_length = 4000         # 最大上下文长度
```

### MCP工具配置

```toml
[mcp]
enabled = true                    # 是否启用MCP工具
servers = []                      # MCP服务器列表
```

### Ascii2D配置

```toml
[ascii2d]
base_url = "https://ascii2d.net"  # Ascii2D基础URL
timeout = 30                      # 请求超时时间
max_results = 10                  # 最大搜索结果数量
```

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

### 扩展RAG功能

1. **添加新的嵌入模型**:
   - 在 `EmbeddingModel` 类中添加新模型支持
   - 实现对应的编码方法
   - 更新配置选项

2. **自定义知识提取**:
   - 在 `KnowledgeExtractor` 中添加新的提取模式
   - 实现特定领域的知识识别
   - 调整重要性评分算法

3. **优化检索算法**:
   - 修改相似度计算方法
   - 调整检索阈值和参数
   - 实现多模态检索

## 🐛 故障排除

### 常见问题

1. **配置错误**: 检查 `config.toml` 文件中的配置是否正确
2. **API密钥无效**: 确认OpenAI API密钥和Telegram Bot Token有效
3. **网络问题**: 检查网络连接和代理设置
4. **权限问题**: 确认机器人有足够的权限
5. **图片搜索失败**: 检查PicImageSearch库是否正确安装
6. **RAG功能异常**:
   - 检查数据库是否正确初始化
   - 确认语义模型下载完成
   - 检查向量存储目录权限
   - 验证嵌入模型配置正确
7. **内存不足**: 语义模型需要较大内存，建议至少2GB
8. **模型下载失败**: 首次运行需要网络连接下载模型

### 日志查看

日志文件保存在 `logs/` 目录下，可以通过查看日志来诊断问题。

### 依赖问题

如果遇到依赖问题，可以尝试：

```bash
# 清理并重新安装（推荐）
uv clean
uv sync

# 或者使用pip
pip install -r requirements.txt
```

### RAG相关问题

```bash
# 重建向量索引
uv run python -c "
import asyncio
from src.services.database_service import DatabaseService
from src.services.rag_service import RAGService

async def rebuild():
    db = DatabaseService()
    rag = RAGService(db)
    await rag.initialize()
    await rag.rebuild_index()

asyncio.run(rebuild())
"

# 检查数据库状态
sqlite3 data/bot.db "SELECT COUNT(*) FROM knowledge_entries;"

# 清理向量存储（重新开始）
rm -rf data/vector_store/*
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

- [OpenAI](https://openai.com/) - 提供AI对话能力
- [Sentence Transformers](https://www.sbert.net/) - 提供强大的语义嵌入模型
- [FAISS](https://faiss.ai/) - 高效的向量相似性搜索
- [python-telegram-bot](https://python-telegram-bot.org/) - Telegram Bot API的Python实现
- [PicImageSearch](https://pic-image-search.kituin.fun/) - 提供强大的图片搜索功能
- [scikit-learn](https://scikit-learn.org/) - 机器学习工具包
- [SQLite](https://www.sqlite.org/) - 轻量级数据库引擎

## 📊 性能和资源使用

### 系统要求

- **内存**: 至少2GB RAM（语义模型需要）
- **存储**: 至少1GB可用空间
- **CPU**: 支持AVX指令集（现代CPU）
- **网络**: 稳定的互联网连接

### 资源使用情况

| 功能 | 内存占用 | 磁盘占用 | 启动时间 |
|------|----------|----------|----------|
| 基础功能 | ~100MB | ~50MB | ~2秒 |
| + RAG (TF-IDF) | ~150MB | ~100MB | ~5秒 |
| + RAG (语义模型) | ~800MB | ~500MB | ~15秒 |

### 优化建议

1. **生产环境**: 使用语义模型获得最佳效果
2. **资源受限**: 使用TF-IDF模型降低资源消耗
3. **批量处理**: 定期清理旧的对话记录和知识条目
4. **监控**: 定期检查数据库大小和向量存储使用情况

---

**注意**: 请确保遵守OpenAI和Telegram的使用条款，合理使用API资源。