"""
知识提取器模块
从对话中智能提取和组织知识
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from loguru import logger

from ..models import Message
from .openai_service import OpenAIService


class KnowledgeExtractor:
    """知识提取器类"""
    
    def __init__(self, openai_service: OpenAIService = None):
        self.openai_service = openai_service
        self.extraction_patterns = self._initialize_patterns()
        
        # 配置参数 - 降低阈值提高敏感度
        self.min_content_length = 15      # 降低最小内容长度
        self.max_content_length = 2000    # 最大内容长度
        self.importance_threshold = 0.05  # 降低重要性阈值
        
        logger.info("知识提取器初始化完成")
    
    def _initialize_patterns(self) -> Dict[str, Any]:
        """初始化提取模式"""
        return {
            # 问答模式
            "qa_patterns": [
                r"(.+?)是什么[？?]?\s*(.+)",
                r"什么是(.+?)[？?]?\s*(.+)",
                r"(.+?)怎么(.+?)[？?]?\s*(.+)",
                r"如何(.+?)[？?]?\s*(.+)",
                r"为什么(.+?)[？?]?\s*(.+)",
            ],
            
            # 定义模式
            "definition_patterns": [
                r"(.+?)是(.+)",
                r"(.+?)指的是(.+)",
                r"(.+?)意思是(.+)",
                r"(.+?)就是(.+)",
            ],
            
            # 步骤模式
            "step_patterns": [
                r"第[一二三四五六七八九十\d]+步[：:](.+)",
                r"步骤[一二三四五六七八九十\d]+[：:](.+)",
                r"\d+[\.、](.+)",
            ],
            
            # 重要信息模式
            "important_patterns": [
                r"重要的是(.+)",
                r"需要注意(.+)",
                r"关键是(.+)",
                r"核心是(.+)",
                r"主要(.+)",
            ],
            
            # 事实模式
            "fact_patterns": [
                r"事实上(.+)",
                r"实际上(.+)",
                r"根据(.+)",
                r"据(.+)显示",
            ]
        }
    
    async def extract_from_conversation(self, messages: List[Message], 
                                      user_id: int, conversation_id: int) -> List[Dict[str, Any]]:
        """从对话中提取知识"""
        try:
            knowledge_items = []
            
            # 分析消息序列
            for i, message in enumerate(messages):
                if message.role == "assistant":
                    # 从助手回复中提取知识
                    items = await self._extract_from_assistant_message(
                        message, user_id, conversation_id, i
                    )
                    knowledge_items.extend(items)
                elif message.role == "user":
                    # 从用户消息中提取个人信息
                    personal_items = self._extract_personal_info(
                        message.content, user_id, conversation_id, i
                    )
                    knowledge_items.extend(personal_items)
                
                # 分析对话对（用户问题+助手回答）
                if (i > 0 and messages[i-1].role == "user" and 
                    message.role == "assistant"):
                    qa_item = await self._extract_qa_pair(
                        messages[i-1], message, user_id, conversation_id
                    )
                    if qa_item:
                        knowledge_items.append(qa_item)
            
            # 使用LLM进行高级知识提取（如果可用）
            if self.openai_service and len(messages) >= 2:
                llm_items = await self._extract_with_llm(
                    messages, user_id, conversation_id
                )
                knowledge_items.extend(llm_items)
            
            # 去重和过滤
            knowledge_items = self._deduplicate_and_filter(knowledge_items)
            
            logger.info(f"从 {len(messages)} 条消息中提取了 {len(knowledge_items)} 个知识项")
            return knowledge_items
            
        except Exception as e:
            logger.error(f"从对话提取知识失败: {e}")
            return []
    
    async def _extract_from_assistant_message(self, message: Message, 
                                            user_id: int, conversation_id: int, 
                                            message_index: int) -> List[Dict[str, Any]]:
        """从助手消息中提取知识"""
        knowledge_items = []
        content = message.content.strip()
        
        if len(content) < self.min_content_length:
            return knowledge_items
        
        try:
            # 使用规则提取
            items = self._extract_with_rules(content, user_id, conversation_id, message_index)
            knowledge_items.extend(items)
            
            # 提取列表和步骤
            list_items = self._extract_lists_and_steps(content, user_id, conversation_id, message_index)
            knowledge_items.extend(list_items)
            
            # 提取定义和解释
            definition_items = self._extract_definitions(content, user_id, conversation_id, message_index)
            knowledge_items.extend(definition_items)
            
            # 提取个人信息
            personal_items = self._extract_personal_info(content, user_id, conversation_id, message_index)
            knowledge_items.extend(personal_items)
            
        except Exception as e:
            logger.error(f"从助手消息提取知识失败: {e}")
        
        return knowledge_items
    
    def _extract_with_rules(self, content: str, user_id: int, 
                          conversation_id: int, message_index: int) -> List[Dict[str, Any]]:
        """使用规则提取知识"""
        knowledge_items = []
        
        # 按句子分割内容
        sentences = re.split(r'[。！？\n]', content)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # 句子太短
                continue
            
            # 检查重要信息模式
            for pattern in self.extraction_patterns["important_patterns"]:
                match = re.search(pattern, sentence)
                if match:
                    knowledge_items.append({
                        'title': f"重要信息: {sentence[:30]}",
                        'content': sentence,
                        'category': 'important',
                        'importance_score': 0.8,
                        'keywords': self._extract_keywords(sentence),
                        'user_id': user_id,
                        'conversation_id': conversation_id,
                        'source_info': f"message_{message_index}"
                    })
                    break
            
            # 检查事实模式
            for pattern in self.extraction_patterns["fact_patterns"]:
                match = re.search(pattern, sentence)
                if match:
                    knowledge_items.append({
                        'title': f"事实: {sentence[:30]}",
                        'content': sentence,
                        'category': 'fact',
                        'importance_score': 0.7,
                        'keywords': self._extract_keywords(sentence),
                        'user_id': user_id,
                        'conversation_id': conversation_id,
                        'source_info': f"message_{message_index}"
                    })
                    break
        
        return knowledge_items
    
    def _extract_lists_and_steps(self, content: str, user_id: int, 
                                conversation_id: int, message_index: int) -> List[Dict[str, Any]]:
        """提取列表和步骤"""
        knowledge_items = []
        
        # 查找步骤模式
        steps = []
        for pattern in self.extraction_patterns["step_patterns"]:
            matches = re.findall(pattern, content)
            steps.extend(matches)
        
        if len(steps) >= 2:  # 至少有2个步骤才认为是有价值的
            step_content = "\n".join([f"{i+1}. {step}" for i, step in enumerate(steps)])
            knowledge_items.append({
                'title': f"操作步骤: {steps[0][:20]}...",
                'content': step_content,
                'category': 'procedure',
                'importance_score': 0.9,
                'keywords': self._extract_keywords(step_content),
                'user_id': user_id,
                'conversation_id': conversation_id,
                'source_info': f"message_{message_index}"
            })
        
        # 查找编号列表
        numbered_list = re.findall(r'\d+[\.、](.+)', content)
        if len(numbered_list) >= 3:
            list_content = "\n".join([f"{i+1}. {item}" for i, item in enumerate(numbered_list)])
            knowledge_items.append({
                'title': f"列表: {numbered_list[0][:20]}...",
                'content': list_content,
                'category': 'list',
                'importance_score': 0.6,
                'keywords': self._extract_keywords(list_content),
                'user_id': user_id,
                'conversation_id': conversation_id,
                'source_info': f"message_{message_index}"
            })
        
        return knowledge_items
    
    def _extract_definitions(self, content: str, user_id: int, 
                           conversation_id: int, message_index: int) -> List[Dict[str, Any]]:
        """提取定义和解释"""
        knowledge_items = []
        
        for pattern in self.extraction_patterns["definition_patterns"]:
            matches = re.findall(pattern, content)
            for match in matches:
                if len(match) == 2:
                    term, definition = match
                    term = term.strip()
                    definition = definition.strip()
                    
                    if len(term) > 1 and len(definition) > 10:
                        knowledge_items.append({
                            'title': f"定义: {term}",
                            'content': f"{term}是{definition}",
                            'category': 'definition',
                            'importance_score': 0.8,
                            'keywords': self._extract_keywords(f"{term} {definition}"),
                            'user_id': user_id,
                            'conversation_id': conversation_id,
                            'source_info': f"message_{message_index}"
                        })
        
        return knowledge_items
    
    def _extract_personal_info(self, content: str, user_id: int, 
                              conversation_id: int, message_index: int) -> List[Dict[str, Any]]:
        """提取个人信息"""
        knowledge_items = []
        
        # 个人信息模式 - 更精确的匹配
        personal_patterns = [
            (r'我的(.{1,10})是(.{1,50})', 'attr_value'),  # "我的X是Y"
            (r'我的生日是?(.{1,20})', 'birthday'),  # "我的生日是X"
            (r'我叫(.{1,20})', 'name'),  # "我叫X"
            (r'我是(.{1,50})', 'identity'),  # "我是X" 
            (r'我在(.{1,30})', 'location'),  # "我在X"
            (r'我来自(.{1,30})', 'origin'),  # "我来自X"
            (r'我住在(.{1,30})', 'residence'),  # "我住在X"
            (r'我喜欢(.{1,50})', 'likes'),  # "我喜欢X"
            (r'我不喜欢(.{1,50})', 'dislikes'),  # "我不喜欢X"
            (r'我(.{1,10})岁', 'age'),  # "我X岁"
            (r'我是(.{1,10})年(.{1,10})月(.{1,10})日生的', 'full_birthday')  # 完整生日
        ]
        
        for pattern, info_type in personal_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                item = self._create_personal_info_item(
                    match, info_type, user_id, conversation_id, message_index
                )
                if item:
                    knowledge_items.append(item)
        
        return knowledge_items
    
    def _create_personal_info_item(self, match, info_type: str, user_id: int, 
                                  conversation_id: int, message_index: int) -> Optional[Dict[str, Any]]:
        """创建个人信息知识项"""
        try:
            if info_type == 'attr_value' and isinstance(match, tuple) and len(match) == 2:
                attr, value = match
                if len(attr.strip()) > 0 and len(value.strip()) > 1:
                    return {
                        'title': f"个人信息: {attr.strip()}",
                        'content': f"用户的{attr.strip()}是{value.strip()}",
                        'category': 'personal_info',
                        'importance_score': 0.9,
                        'keywords': f"{attr.strip()},{value.strip()}",
                        'user_id': user_id,
                        'conversation_id': conversation_id,
                        'source_info': f"personal_info_{message_index}"
                    }
            
            elif info_type == 'full_birthday' and isinstance(match, tuple) and len(match) == 3:
                year, month, day = match
                return {
                    'title': f"个人信息: 生日",
                    'content': f"用户是{year}年{month}月{day}日生的",
                    'category': 'personal_info',
                    'importance_score': 0.9,
                    'keywords': f"生日,{year}年,{month}月,{day}日",
                    'user_id': user_id,
                    'conversation_id': conversation_id,
                    'source_info': f"personal_info_{message_index}"
                }
            
            elif isinstance(match, str) and len(match.strip()) > 1:
                # 处理单个匹配的情况
                value = match.strip()
                
                # 根据信息类型设置标题和内容
                type_mapping = {
                    'birthday': ('生日', f"用户的生日是{value}"),
                    'name': ('姓名', f"用户叫{value}"),
                    'identity': ('身份', f"用户是{value}"),
                    'location': ('位置', f"用户在{value}"),
                    'origin': ('来源', f"用户来自{value}"),
                    'residence': ('住址', f"用户住在{value}"),
                    'likes': ('喜好', f"用户喜欢{value}"),
                    'dislikes': ('不喜欢', f"用户不喜欢{value}"),
                    'age': ('年龄', f"用户{value}岁")
                }
                
                if info_type in type_mapping:
                    title_suffix, content = type_mapping[info_type]
                    return {
                        'title': f"个人信息: {title_suffix}",
                        'content': content,
                        'category': 'personal_info',
                        'importance_score': 0.9 if info_type in ['birthday', 'name'] else 0.8,
                        'keywords': f"{title_suffix},{value}",
                        'user_id': user_id,
                        'conversation_id': conversation_id,
                        'source_info': f"personal_info_{message_index}"
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"创建个人信息项失败: {e}")
            return None
    
    async def _extract_qa_pair(self, user_message: Message, assistant_message: Message,
                             user_id: int, conversation_id: int) -> Optional[Dict[str, Any]]:
        """提取问答对"""
        try:
            question = user_message.content.strip()
            answer = assistant_message.content.strip()
            
            # 过滤太短的内容
            if len(question) < 5 or len(answer) < 20:
                return None
            
            # 检查是否是问题或个人信息分享
            question_indicators = ['？', '?', '什么', '如何', '怎么', '为什么', '哪里', '什么时候']
            personal_info_indicators = ['我的', '我是', '我叫', '我在', '我来自', '我住在', '我喜欢', '我不喜欢']
            
            is_question = any(indicator in question for indicator in question_indicators)
            is_personal_info = any(indicator in question for indicator in personal_info_indicators)
            
            if not (is_question or is_personal_info):
                return None
            
            # 计算重要性分数
            importance_score = self._calculate_qa_importance(question, answer, is_personal_info)
            
            if importance_score < self.importance_threshold:
                return None
            
            # 根据内容类型设置标题和类别
            if is_personal_info:
                title = f"个人信息: {question[:30]}"
                category = 'personal_info'
            else:
                title = f"问答: {question[:30]}"
                category = 'qa'
            
            return {
                'title': title,
                'content': f"用户: {question}\n\n回复: {answer}",
                'category': category,
                'importance_score': importance_score,
                'keywords': self._extract_keywords(f"{question} {answer}"),
                'user_id': user_id,
                'conversation_id': conversation_id,
                'source_info': 'qa_pair'
            }
            
        except Exception as e:
            logger.error(f"提取问答对失败: {e}")
            return None
    
    def _calculate_qa_importance(self, question: str, answer: str, is_personal_info: bool = False) -> float:
        """计算问答对的重要性分数"""
        score = 0.0
        
        # 基础分数
        if is_personal_info:
            score += 0.6  # 个人信息给予更高的基础分数
        else:
            score += 0.4
        
        # 答案长度加分 (降低门槛)
        if len(answer) > 50:
            score += 0.3
        elif len(answer) > 20:
            score += 0.2
        elif len(answer) > 10:
            score += 0.1
        
        # 个人信息类型加分
        if is_personal_info:
            personal_keywords = {
                '生日': 0.5, '年龄': 0.4, '姓名': 0.5, '名字': 0.5,
                '职业': 0.4, '工作': 0.4, '学校': 0.4, '专业': 0.4,
                '家乡': 0.3, '住址': 0.3, '电话': 0.4, '邮箱': 0.4,
                '爱好': 0.3, '兴趣': 0.3, '喜欢': 0.2, '不喜欢': 0.2
            }
            
            for keyword, bonus in personal_keywords.items():
                if keyword in question or keyword in answer:
                    score += bonus
                    break  # 只加一次分数
        
        # 问题类型加分 (提高权重)
        if any(word in question for word in ['如何', '怎么', '步骤', '怎样']):
            score += 0.4  # 操作类问题
        elif any(word in question for word in ['什么是', '定义', '概念', '是什么']):
            score += 0.3  # 定义类问题
        elif any(word in question for word in ['为什么', '原因', '原理', '为啥']):
            score += 0.3  # 解释类问题
        elif any(word in question for word in ['什么', '哪个', '哪里', '什么时候']):
            score += 0.2  # 一般疑问
        
        # 答案结构化程度加分
        if any(pattern in answer for pattern in ['1.', '2.', '第一', '第二', '首先', '其次', '然后', '最后']):
            score += 0.3  # 结构化答案
        
        # 答案包含关键信息加分
        if any(word in answer for word in ['重要', '关键', '核心', '主要', '注意']):
            score += 0.2
        
        return min(score, 1.0)
    
    async def _extract_with_llm(self, messages: List[Message], 
                              user_id: int, conversation_id: int) -> List[Dict[str, Any]]:
        """使用LLM进行高级知识提取"""
        try:
            if not self.openai_service:
                return []
            
            # 构建对话文本
            conversation_text = self._build_conversation_text(messages)
            if len(conversation_text) > 4000:  # 限制长度
                conversation_text = conversation_text[:4000]
            
            # 构建提取提示
            extraction_prompt = f"""
请从以下对话中提取有价值的知识点。对于每个知识点，请提供：
1. 标题（简洁描述）
2. 内容（详细描述）
3. 类别（definition/procedure/fact/tip/other）
4. 重要性评分（0-1之间的小数）
5. 关键词（用逗号分隔）

对话内容：
{conversation_text}

请以JSON格式返回结果，格式如下：
[
    {{
        "title": "知识点标题",
        "content": "知识点详细内容",
        "category": "类别",
        "importance_score": 0.8,
        "keywords": ["关键词1", "关键词2"]
    }}
]

只提取真正有价值的知识点，避免重复和琐碎信息。
"""
            
            # 调用LLM
            response = await self.openai_service.chat_completion(
                messages=[{"role": "user", "content": extraction_prompt}],
                tools=None
            )
            
            if response and isinstance(response, str):
                result_text = response.strip()
                
                # 尝试解析JSON
                try:
                    extracted_items = json.loads(result_text)
                    
                    # 添加用户和对话信息
                    for item in extracted_items:
                        item['user_id'] = user_id
                        item['conversation_id'] = conversation_id
                        item['source_info'] = 'llm_extraction'
                        
                        # 确保关键词是字符串
                        if isinstance(item.get('keywords'), list):
                            item['keywords'] = ','.join(item['keywords'])
                    
                    logger.info(f"LLM提取了 {len(extracted_items)} 个知识点")
                    return extracted_items
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"LLM返回的JSON解析失败: {e}")
                    return []
            
            return []
            
        except Exception as e:
            logger.error(f"LLM知识提取失败: {e}")
            return []
    
    def _build_conversation_text(self, messages: List[Message]) -> str:
        """构建对话文本"""
        conversation_parts = []
        
        for message in messages[-10:]:  # 只取最后10条消息
            role_name = "用户" if message.role == "user" else "助手"
            conversation_parts.append(f"{role_name}: {message.content}")
        
        return "\n\n".join(conversation_parts)
    
    def _extract_keywords(self, text: str, max_keywords: int = 5) -> str:
        """提取关键词"""
        # 简单的关键词提取
        words = re.findall(r'\b\w{2,}\b', text.lower())
        
        # 过滤常见词
        stop_words = {
            '的', '是', '了', '在', '有', '和', '与', '或', '但', '而', '就', '都',
            '也', '还', '只', '又', '很', '更', '最', '非常', '特别', '比较',
            '可以', '能够', '需要', '应该', '必须', '会', '要', '想', '知道',
            '这个', '那个', '这些', '那些', '什么', '如何', '为什么', '怎么'
        }
        
        keywords = [word for word in words if word not in stop_words and len(word) > 1]
        
        # 统计词频
        word_count = {}
        for word in keywords:
            word_count[word] = word_count.get(word, 0) + 1
        
        # 按频率排序，取前几个
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        top_keywords = [word for word, count in sorted_words[:max_keywords]]
        
        return ','.join(top_keywords)
    
    def _deduplicate_and_filter(self, knowledge_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """去重和过滤知识项"""
        if not knowledge_items:
            return []
        
        # 按内容去重
        seen_contents = set()
        unique_items = []
        
        for item in knowledge_items:
            content_hash = hash(item['content'][:100])  # 使用前100个字符的hash
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_items.append(item)
        
        # 过滤低质量项目
        filtered_items = []
        for item in unique_items:
            # 检查长度
            if len(item['content']) < self.min_content_length:
                continue
            
            # 检查重要性分数
            if item.get('importance_score', 0) < self.importance_threshold:
                continue
            
            # 检查内容质量
            if self._is_low_quality_content(item['content']):
                continue
            
            filtered_items.append(item)
        
        # 按重要性分数排序
        filtered_items.sort(key=lambda x: x.get('importance_score', 0), reverse=True)
        
        return filtered_items
    
    def _is_low_quality_content(self, content: str) -> bool:
        """检查是否是低质量内容"""
        # 检查重复字符
        if len(set(content)) < len(content) * 0.3:
            return True
        
        # 检查是否主要是标点符号
        punctuation_count = sum(1 for char in content if not char.isalnum() and not char.isspace())
        if punctuation_count > len(content) * 0.5:
            return True
        
        # 检查是否包含有意义的内容
        meaningful_words = re.findall(r'\b\w{3,}\b', content)
        if len(meaningful_words) < 3:
            return True
        
        return False
    
    async def extract_single_knowledge(self, content: str, user_id: int, 
                                     conversation_id: int, category: str = "manual") -> Optional[Dict[str, Any]]:
        """提取单个知识项（用于手动添加）"""
        try:
            if len(content) < self.min_content_length:
                return None
            
            # 生成标题
            title = content[:50].strip()
            if not title:
                return None
            
            # 提取关键词
            keywords = self._extract_keywords(content)
            
            # 计算重要性分数
            importance_score = min(len(content) / 200.0, 1.0)
            
            return {
                'title': title,
                'content': content,
                'category': category,
                'importance_score': importance_score,
                'keywords': keywords,
                'user_id': user_id,
                'conversation_id': conversation_id,
                'source_info': 'manual'
            }
            
        except Exception as e:
            logger.error(f"提取单个知识项失败: {e}")
            return None
