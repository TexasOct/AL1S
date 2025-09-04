"""
RAG (检索增强生成) 服务模块
提供向量存储、检索和嵌入功能，让bot能够从对话中学习和积累知识
"""

import json
import uuid
import pickle
import asyncio
import re
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
from loguru import logger

try:
    import faiss
except ImportError:
    logger.warning("faiss-cpu 未安装，RAG功能将不可用")
    faiss = None

from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("sentence-transformers 未安装，将使用TF-IDF模型")
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from ..config import config
from .database_service import DatabaseService


class EmbeddingModel:
    """嵌入模型接口"""
    
    def __init__(self, model_name: str = "tfidf"):
        self.model_name = model_name
        self.dimension = 0
        self.model = None
        self._fitted = False
        
        if model_name == "tfidf":
            self._init_tfidf_model()
        elif model_name.startswith("sentence-transformers"):
            self._init_sentence_transformer_model(model_name)
        else:
            raise ValueError(f"不支持的嵌入模型: {model_name}")
    
    def _init_tfidf_model(self):
        """初始化TF-IDF模型"""
        self.vectorizer = TfidfVectorizer(
            max_features=1024,  # 增加特征数量提高精度
            stop_words=None,    # 中文没有标准停用词
            ngram_range=(1, 3), # 扩展到3-gram提高语义捕获
            min_df=1,           # 最小文档频率
            max_df=0.8,         # 降低最大文档频率，过滤高频无意义词
            token_pattern=r'(?u)\b\w+\b',  # 优化中文分词
            lowercase=True,     # 转换为小写
            sublinear_tf=True,  # 使用对数TF权重
            norm='l2'          # L2归一化
        )
        self.dimension = 1024
        logger.info("使用TF-IDF嵌入模型")
    
    def _init_sentence_transformer_model(self, model_name: str):
        """初始化Sentence Transformer模型"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ValueError("sentence-transformers 未安装，无法使用语义嵌入模型")
        
        # 从模型名称中提取实际的模型路径
        model_path = model_name.replace("sentence-transformers/", "")
        
        try:
            self.model = SentenceTransformer(model_path)
            self.dimension = self.model.get_sentence_embedding_dimension()
            self._fitted = True  # Sentence Transformers 不需要训练
            logger.info(f"使用Sentence Transformers模型: {model_path}, 维度: {self.dimension}")
        except Exception as e:
            logger.error(f"加载Sentence Transformers模型失败: {e}")
            # 回退到TF-IDF
            logger.info("回退到TF-IDF模型")
            self.model_name = "tfidf"
            self._init_tfidf_model()
    
    def fit(self, texts: List[str]) -> None:
        """训练模型（仅TF-IDF需要）"""
        if self.model_name == "tfidf" and not self._fitted:
            if texts:
                # 预处理训练文本
                processed_texts = [self._preprocess_text(text) for text in texts]
                self.vectorizer.fit(processed_texts)
                self._fitted = True
                logger.info(f"TF-IDF模型训练完成，使用 {len(texts)} 个文档")
        elif self.model_name.startswith("sentence-transformers"):
            # Sentence Transformers 不需要训练
            logger.debug("Sentence Transformers模型无需训练")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """将文本编码为向量"""
        if self.model_name == "tfidf":
            if not self._fitted:
                # 如果模型未训练，先用当前文本训练
                self.fit(texts)
            
            # 预处理所有文本
            processed_texts = [self._preprocess_text(text) for text in texts]
            vectors = self.vectorizer.transform(processed_texts)
            return vectors.toarray().astype(np.float32)
        elif self.model_name.startswith("sentence-transformers"):
            # 使用Sentence Transformers编码
            vectors = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return vectors.astype(np.float32)
        
        return np.array([])
    
    def encode_single(self, text: str) -> np.ndarray:
        """编码单个文本"""
        if self.model_name == "tfidf":
            if not self._fitted:
                logger.warning("TF-IDF模型未训练，返回零向量")
                return np.zeros(self.dimension, dtype=np.float32)
            
            # 预处理文本
            processed_text = self._preprocess_text(text)
            return self.encode([processed_text])[0]
        elif self.model_name.startswith("sentence-transformers"):
            # 使用Sentence Transformers编码单个文本
            vector = self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
            return vector.astype(np.float32)
        
        return np.zeros(self.dimension, dtype=np.float32)
    
    def _preprocess_text(self, text: str) -> str:
        """预处理中文文本，提高检索敏感度"""
        if not text:
            return text
        
        # 1. 规范化空白字符
        text = re.sub(r'\s+', ' ', text.strip())
        
        # 2. 移除特殊符号但保留中文标点
        text = re.sub(r'[^\w\s\u4e00-\u9fff，。！？；：""''（）【】《》、]', ' ', text)
        
        # 3. 分离中英文，在中英文之间添加空格
        text = re.sub(r'([\u4e00-\u9fff])([a-zA-Z0-9])', r'\1 \2', text)
        text = re.sub(r'([a-zA-Z0-9])([\u4e00-\u9fff])', r'\1 \2', text)
        
        # 4. 处理数字和单位
        text = re.sub(r'(\d+)([a-zA-Z\u4e00-\u9fff])', r'\1 \2', text)
        
        # 5. 再次规范化空白字符
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text


class KnowledgeEntry:
    """知识条目类"""
    
    def __init__(self, id: int = None, user_id: int = None, conversation_id: int = None,
                 title: str = "", content: str = "", summary: str = "", 
                 keywords: str = "", category: str = "general", 
                 importance_score: float = 0.0, embedding_id: str = None,
                 source_message_id: int = None, created_at: datetime = None):
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


class RAGService:
    """RAG服务类"""
    
    def __init__(self, database_service: DatabaseService, 
                 vector_store_path: str = "data/vector_store"):
        self.database_service = database_service
        self.vector_store_path = Path(vector_store_path)
        self.vector_store_path.mkdir(exist_ok=True)
        
        # 初始化嵌入模型
        self.embedding_model = EmbeddingModel(config.rag.embedding_model)
        
        # 初始化FAISS索引
        self.index = None
        self.knowledge_id_map = {}  # FAISS索引ID到知识条目ID的映射
        
        # 向量索引文件路径（包含模型名称以避免冲突）
        model_name_safe = config.rag.embedding_model.replace("/", "_").replace("-", "_")
        self.index_file = self.vector_store_path / f"faiss_index_{model_name_safe}.bin"
        self.id_map_file = self.vector_store_path / f"id_mapping_{model_name_safe}.json"
        self.vectorizer_file = self.vector_store_path / f"vectorizer_{model_name_safe}.pkl"
        
        # 配置参数 - 使用配置文件中的设置
        self.max_knowledge_entries = config.rag.max_knowledge_entries
        self.similarity_threshold = config.rag.similarity_threshold
        self.top_k_retrieval = config.rag.top_k_retrieval
        
        logger.info("RAG服务初始化完成")
    
    async def initialize(self) -> bool:
        """初始化RAG服务"""
        try:
            # 先尝试加载持久化的向量索引
            if await self._load_persistent_index():
                logger.info("成功加载持久化的向量索引")
            else:
                # 如果没有持久化索引，从数据库重建
                logger.info("未找到持久化索引，从数据库重建")
                await self._load_knowledge_base()
            
            logger.info("RAG服务初始化成功")
            return True
        except Exception as e:
            logger.error(f"RAG服务初始化失败: {e}")
            return False
    
    async def _load_knowledge_base(self) -> None:
        """从数据库加载知识库到向量存储"""
        try:
            # 获取所有知识条目
            knowledge_entries = await self._get_all_knowledge_entries()
            
            if not knowledge_entries:
                logger.info("没有找到现有的知识条目")
                return
            
            # 提取文本内容
            texts = [entry.content for entry in knowledge_entries]
            
            # 训练嵌入模型
            self.embedding_model.fit(texts)
            
            # 生成向量
            vectors = self.embedding_model.encode(texts)
            
            # 初始化FAISS索引
            if faiss is not None and len(vectors) > 0:
                dimension = vectors.shape[1]
                self.index = faiss.IndexFlatIP(dimension)  # 使用内积索引
                
                # 归一化向量（用于余弦相似度）
                faiss.normalize_L2(vectors)
                
                # 添加向量到索引
                self.index.add(vectors)
                
                # 建立ID映射
                self.knowledge_id_map = {i: entry.id for i, entry in enumerate(knowledge_entries)}
                
                logger.info(f"加载了 {len(knowledge_entries)} 个知识条目到向量存储")
                
                # 保存向量索引到磁盘
                await self._save_persistent_index()
            else:
                logger.warning("FAISS不可用或没有有效向量，使用备用检索方法")
                
        except Exception as e:
            logger.error(f"加载知识库失败: {e}")
    
    async def _save_persistent_index(self) -> bool:
        """保存向量索引到磁盘"""
        try:
            if self.index is None or not self.knowledge_id_map:
                return False
            
            # 保存FAISS索引
            if faiss is not None:
                faiss.write_index(self.index, str(self.index_file))
            
            # 保存ID映射
            with open(self.id_map_file, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge_id_map, f, ensure_ascii=False, indent=2)
            
            # 保存模型相关数据
            if self.embedding_model._fitted:
                model_data = {
                    'model_name': self.embedding_model.model_name,
                    'dimension': self.embedding_model.dimension
                }
                
                if self.embedding_model.model_name == "tfidf":
                    model_data['vectorizer'] = self.embedding_model.vectorizer
                # Sentence Transformers 不需要保存模型本身，只保存元数据
                
                with open(self.vectorizer_file, 'wb') as f:
                    pickle.dump(model_data, f)
            
            logger.info(f"向量索引已保存到: {self.vector_store_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存向量索引失败: {e}")
            return False
    
    async def _load_persistent_index(self) -> bool:
        """从磁盘加载向量索引"""
        try:
            # 检查文件是否存在
            if not (self.index_file.exists() and self.id_map_file.exists()):
                return False
            
            # 加载FAISS索引
            if faiss is not None:
                self.index = faiss.read_index(str(self.index_file))
            else:
                return False
            
            # 加载ID映射
            with open(self.id_map_file, 'r', encoding='utf-8') as f:
                # JSON中的键是字符串，需要转换为整数
                id_map_str = json.load(f)
                self.knowledge_id_map = {int(k): v for k, v in id_map_str.items()}
            
            # 加载模型相关数据
            if self.vectorizer_file.exists():
                with open(self.vectorizer_file, 'rb') as f:
                    model_data = pickle.load(f)
                    
                # 检查模型兼容性
                if model_data.get('model_name') != self.embedding_model.model_name:
                    logger.warning(f"模型类型不匹配: 文件中为 {model_data.get('model_name')}, 当前为 {self.embedding_model.model_name}")
                    return False
                
                if self.embedding_model.model_name == "tfidf":
                    self.embedding_model.vectorizer = model_data['vectorizer']
                    self.embedding_model._fitted = True
                # Sentence Transformers 模型在初始化时已经加载
            
            logger.info(f"从磁盘加载了 {self.index.ntotal} 个向量")
            return True
            
        except Exception as e:
            logger.error(f"加载持久化索引失败: {e}")
            return False
    
    async def _get_all_knowledge_entries(self) -> List[KnowledgeEntry]:
        """获取所有知识条目"""
        try:
            with self.database_service.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT id, user_id, conversation_id, title, content, summary,
                           keywords, category, importance_score, embedding_id,
                           source_message_id, created_at
                    FROM knowledge_entries
                    ORDER BY importance_score DESC, created_at DESC
                    LIMIT ?
                """, (self.max_knowledge_entries,))
                
                entries = []
                for row in cursor.fetchall():
                    entry = KnowledgeEntry(
                        id=row['id'],
                        user_id=row['user_id'],
                        conversation_id=row['conversation_id'],
                        title=row['title'],
                        content=row['content'],
                        summary=row['summary'],
                        keywords=row['keywords'],
                        category=row['category'],
                        importance_score=row['importance_score'],
                        embedding_id=row['embedding_id'],
                        source_message_id=row['source_message_id'],
                        created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None
                    )
                    entries.append(entry)
                
                return entries
                
        except Exception as e:
            logger.error(f"获取知识条目失败: {e}")
            return []
    
    async def extract_knowledge(self, user_id: int, conversation_id: int, 
                              messages: List[str], source_message_id: int = None) -> bool:
        """从对话消息中提取知识"""
        try:
            # 合并消息内容
            combined_content = "\n".join(messages)
            
            # 简单的知识提取逻辑（可以后续用LLM优化）
            if len(combined_content.strip()) < 20:  # 内容太短，不提取
                return False
            
            # 生成标题（取前30个字符）
            title = combined_content[:30].strip()
            if not title:
                return False
            
            # 计算重要性分数（基于长度和关键词）
            importance_score = min(len(combined_content) / 100.0, 1.0)
            
            # 提取关键词（简单实现）
            keywords = self._extract_keywords(combined_content)
            
            # 创建知识条目
            knowledge_entry = KnowledgeEntry(
                user_id=user_id,
                conversation_id=conversation_id,
                title=title,
                content=combined_content,
                summary=title,  # 简单使用标题作为摘要
                keywords=",".join(keywords),
                category="conversation",
                importance_score=importance_score,
                source_message_id=source_message_id
            )
            
            # 保存到数据库
            entry_id = await self._save_knowledge_entry(knowledge_entry)
            if entry_id:
                knowledge_entry.id = entry_id
                
                # 生成向量嵌入
                await self._generate_embedding(knowledge_entry)
                
                logger.info(f"提取并保存知识条目: {title}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"提取知识失败: {e}")
            return False
    
    def _extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """简单的关键词提取"""
        # 这里使用简单的方法，可以后续优化
        words = text.split()
        # 过滤短词和常见词
        filtered_words = [
            word for word in words 
            if len(word) > 2 and word not in ['的', '是', '了', '在', '有', '和', '与', '或']
        ]
        
        # 返回前几个词作为关键词
        return filtered_words[:max_keywords]
    
    async def _save_knowledge_entry(self, entry: KnowledgeEntry) -> Optional[int]:
        """保存知识条目到数据库"""
        try:
            embedding_id = str(uuid.uuid4())
            entry.embedding_id = embedding_id
            
            with self.database_service.get_connection() as conn:
                cursor = conn.execute("""
                    INSERT INTO knowledge_entries 
                    (user_id, conversation_id, title, content, summary, keywords,
                     category, importance_score, embedding_id, source_message_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.user_id, entry.conversation_id, entry.title, entry.content,
                    entry.summary, entry.keywords, entry.category, entry.importance_score,
                    entry.embedding_id, entry.source_message_id
                ))
                
                return cursor.lastrowid
                
        except Exception as e:
            logger.error(f"保存知识条目失败: {e}")
            return None
    
    async def _generate_embedding(self, entry: KnowledgeEntry) -> bool:
        """为知识条目生成向量嵌入"""
        try:
            # 生成向量
            vector = self.embedding_model.encode_single(entry.content)
            
            # 保存向量到数据库
            vector_data = pickle.dumps(vector)
            
            with self.database_service.get_connection() as conn:
                conn.execute("""
                    INSERT INTO embeddings (id, knowledge_entry_id, vector_data, dimension, model_name)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    entry.embedding_id, entry.id, vector_data, 
                    len(vector), self.embedding_model.model_name
                ))
            
            # 如果FAISS索引存在，添加到索引
            if self.index is not None and faiss is not None:
                # 归一化向量
                vector = vector.reshape(1, -1).astype(np.float32)
                faiss.normalize_L2(vector)
                
                # 添加到索引
                new_id = len(self.knowledge_id_map)
                self.index.add(vector)
                self.knowledge_id_map[new_id] = entry.id
                
                # 保存更新后的索引
                await self._save_persistent_index()
            
            logger.debug(f"为知识条目 {entry.id} 生成向量嵌入")
            return True
            
        except Exception as e:
            logger.error(f"生成向量嵌入失败: {e}")
            return False
    
    async def retrieve_knowledge(self, user_id: int, query: str, 
                               conversation_id: int = None) -> List[Tuple[KnowledgeEntry, float]]:
        """检索相关知识"""
        try:
            # 生成查询向量
            query_vector = self.embedding_model.encode_single(query)
            
            if self.index is not None and faiss is not None:
                # 使用FAISS进行快速检索
                return await self._faiss_retrieve(user_id, query, query_vector, conversation_id)
            else:
                # 使用备用检索方法
                return await self._fallback_retrieve(user_id, query, conversation_id)
                
        except Exception as e:
            logger.error(f"检索知识失败: {e}")
            return []
    
    async def _faiss_retrieve(self, user_id: int, query: str, query_vector: np.ndarray,
                            conversation_id: int = None) -> List[Tuple[KnowledgeEntry, float]]:
        """使用FAISS进行向量检索"""
        try:
            # 归一化查询向量
            query_vector = query_vector.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(query_vector)
            
            # 检索最相似的向量
            scores, indices = self.index.search(query_vector, self.top_k_retrieval)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                # 对FAISS内积结果使用更低的阈值
                adjusted_threshold = max(0.01, self.similarity_threshold * 0.2)  # 进一步降低阈值
                if score < adjusted_threshold:
                    continue
                
                # 获取对应的知识条目ID
                if idx in self.knowledge_id_map:
                    knowledge_id = self.knowledge_id_map[idx]
                    entry = await self._get_knowledge_entry_by_id(knowledge_id)
                    if entry and (entry.user_id == user_id or entry.category == 'public'):
                        results.append((entry, float(score)))
            
            # 记录检索历史（仅当有conversation_id时）
            if results and conversation_id is not None:
                await self._record_retrieval(user_id, conversation_id, query, results)
            
            logger.debug(f"FAISS检索返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"FAISS检索失败: {e}")
            return []
    
    async def _fallback_retrieve(self, user_id: int, query: str, 
                               conversation_id: int = None) -> List[Tuple[KnowledgeEntry, float]]:
        """备用检索方法（基于关键词匹配）"""
        try:
            # 获取用户的知识条目
            entries = await self._get_user_knowledge_entries(user_id)
            
            if not entries:
                return []
            
            # 改进的文本相似度计算
            results = []
            
            # 预处理查询文本
            processed_query = self.embedding_model._preprocess_text(query)
            query_lower = processed_query.lower()
            query_words = set(query_lower.split())
            
            for entry in entries:
                # 预处理知识条目文本
                processed_content = self.embedding_model._preprocess_text(entry.content)
                processed_title = self.embedding_model._preprocess_text(entry.title)
                
                content_lower = processed_content.lower()
                title_lower = processed_title.lower()
                
                # 计算多维度相似度分数
                score = 0.0
                
                # 1. 完整匹配
                if query_lower in content_lower:
                    score += 0.6
                if query_lower in title_lower:
                    score += 0.8  # 标题匹配权重更高
                
                # 2. 关键词交集 (提高权重)
                content_words = set(content_lower.split())
                title_words = set(title_lower.split())
                
                content_intersection = query_words.intersection(content_words)
                title_intersection = query_words.intersection(title_words)
                
                if content_intersection:
                    content_ratio = len(content_intersection) / len(query_words)
                    score += content_ratio * 0.5
                
                if title_intersection:
                    title_ratio = len(title_intersection) / len(query_words)
                    score += title_ratio * 0.7
                
                # 3. 部分匹配 (新增)
                for word in query_words:
                    if len(word) > 2:  # 只考虑长度大于2的词
                        if word in content_lower:
                            score += 0.1
                        if word in title_lower:
                            score += 0.15
                
                # 4. 关键词匹配 (新增)
                if entry.keywords:
                    keywords_lower = entry.keywords.lower()
                    for word in query_words:
                        if word in keywords_lower:
                            score += 0.2
                
                # 降低阈值，提高敏感度
                if score >= max(0.1, self.similarity_threshold * 0.5):
                    results.append((entry, score))
            
            # 按分数排序
            results.sort(key=lambda x: x[1], reverse=True)
            results = results[:self.top_k_retrieval]
            
            # 记录检索历史（仅当有conversation_id时）
            if results and conversation_id is not None:
                await self._record_retrieval(user_id, conversation_id, query, results)
            
            logger.debug(f"备用检索返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"备用检索失败: {e}")
            return []
    
    async def _get_knowledge_entry_by_id(self, entry_id: int) -> Optional[KnowledgeEntry]:
        """根据ID获取知识条目"""
        try:
            with self.database_service.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT id, user_id, conversation_id, title, content, summary,
                           keywords, category, importance_score, embedding_id,
                           source_message_id, created_at
                    FROM knowledge_entries
                    WHERE id = ?
                """, (entry_id,))
                
                row = cursor.fetchone()
                if row:
                    return KnowledgeEntry(
                        id=row['id'],
                        user_id=row['user_id'],
                        conversation_id=row['conversation_id'],
                        title=row['title'],
                        content=row['content'],
                        summary=row['summary'],
                        keywords=row['keywords'],
                        category=row['category'],
                        importance_score=row['importance_score'],
                        embedding_id=row['embedding_id'],
                        source_message_id=row['source_message_id'],
                        created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None
                    )
                
                return None
                
        except Exception as e:
            logger.error(f"获取知识条目失败: {e}")
            return None
    
    async def _get_user_knowledge_entries(self, user_id: int) -> List[KnowledgeEntry]:
        """获取用户的知识条目"""
        try:
            with self.database_service.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT id, user_id, conversation_id, title, content, summary,
                           keywords, category, importance_score, embedding_id,
                           source_message_id, created_at
                    FROM knowledge_entries
                    WHERE user_id = ? OR category = 'public'
                    ORDER BY importance_score DESC, created_at DESC
                    LIMIT ?
                """, (user_id, self.max_knowledge_entries))
                
                entries = []
                for row in cursor.fetchall():
                    entry = KnowledgeEntry(
                        id=row['id'],
                        user_id=row['user_id'],
                        conversation_id=row['conversation_id'],
                        title=row['title'],
                        content=row['content'],
                        summary=row['summary'],
                        keywords=row['keywords'],
                        category=row['category'],
                        importance_score=row['importance_score'],
                        embedding_id=row['embedding_id'],
                        source_message_id=row['source_message_id'],
                        created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None
                    )
                    entries.append(entry)
                
                return entries
                
        except Exception as e:
            logger.error(f"获取用户知识条目失败: {e}")
            return []
    
    async def _record_retrieval(self, user_id: int, conversation_id: int, query: str,
                              results: List[Tuple[KnowledgeEntry, float]]) -> None:
        """记录检索历史"""
        try:
            knowledge_ids = [str(entry.id) for entry, _ in results]
            scores = [score for _, score in results]
            
            with self.database_service.get_connection() as conn:
                conn.execute("""
                    INSERT INTO knowledge_retrievals 
                    (user_id, conversation_id, query, retrieved_knowledge_ids, similarity_scores)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    user_id, conversation_id, query,
                    json.dumps(knowledge_ids), json.dumps(scores)
                ))
            
        except Exception as e:
            logger.error(f"记录检索历史失败: {e}")
    
    async def get_rag_stats(self) -> Dict[str, Any]:
        """获取RAG统计信息"""
        try:
            with self.database_service.get_connection() as conn:
                cursor = conn.execute("SELECT * FROM rag_stats")
                stats = {}
                for row in cursor.fetchall():
                    row_dict = dict(row)
                    stats[row_dict['table_name']] = {
                        'total_count': row_dict['total_count'],
                        'unique_users': row_dict.get('unique_users', 0),
                        'additional_metric': row_dict.get('avg_importance', row_dict.get('avg_dimension', row_dict.get('usage_rate', 0))),
                        'last_created': row_dict['last_created']
                    }
                
                # 添加向量索引信息
                if self.index is not None:
                    stats['vector_index'] = {
                        'total_vectors': self.index.ntotal,
                        'dimension': self.index.d if hasattr(self.index, 'd') else 0,
                        'index_type': type(self.index).__name__
                    }
                
                return stats
                
        except Exception as e:
            logger.error(f"获取RAG统计失败: {e}")
            return {}
    
    async def cleanup_old_knowledge(self, days: int = 90) -> int:
        """清理旧的知识条目"""
        try:
            with self.database_service.get_connection() as conn:
                # 删除旧的知识条目
                cursor = conn.execute("""
                    DELETE FROM knowledge_entries 
                    WHERE created_at < datetime('now', '-{} days')
                    AND importance_score < 0.3
                """.format(days))
                
                deleted_count = cursor.rowcount
                
                # 清理相关的嵌入数据
                conn.execute("""
                    DELETE FROM embeddings 
                    WHERE knowledge_entry_id NOT IN (
                        SELECT id FROM knowledge_entries
                    )
                """)
                
                if deleted_count > 0:
                    # 重新构建向量索引
                    await self._load_knowledge_base()
                    logger.info(f"清理了 {deleted_count} 个旧知识条目")
                
                return deleted_count
                
        except Exception as e:
            logger.error(f"清理旧知识失败: {e}")
            return 0
    
    async def rebuild_index(self) -> bool:
        """重建向量索引"""
        try:
            logger.info("开始重建向量索引...")
            
            # 清除现有索引
            self.index = None
            self.knowledge_id_map = {}
            
            # 重新加载并构建索引
            await self._load_knowledge_base()
            
            logger.info("向量索引重建完成")
            return True
        except Exception as e:
            logger.error(f"重建向量索引失败: {e}")
            return False
