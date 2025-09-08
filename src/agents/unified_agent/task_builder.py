"""
任务构建器 - 简化复杂任务的创建与编排
- 提供声明式任务构建接口
- 支持工作流模板
- 智能任务优化与合并
- 任务链式调用
"""

import asyncio
from typing import Any, Dict, List, Optional, Set, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json

from loguru import logger

from .task_orchestrator import (
    Task, TaskType, TaskOrchestrator, TaskExecutionContext,
    TaskStatus, TaskResult
)


@dataclass
class TaskChain:
    """任务链 - 支持链式任务构建"""
    tasks: List[Task] = field(default_factory=list)
    orchestrator: Optional[TaskOrchestrator] = None
    execution_context: Optional[TaskExecutionContext] = None
    
    def add_task(
        self,
        task_type: TaskType,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        dependencies: Set[str] = None,
        priority: int = 1,
        max_retries: int = 3,
        timeout: float = 300.0
    ) -> 'TaskChain':
        """添加任务到链中"""
        # 自动处理依赖关系 - 新任务依赖链中最后一个任务
        if not dependencies and self.tasks:
            dependencies = {self.tasks[-1].task_id}
        
        task = Task(
            task_id="",  # 将在创建时自动生成
            task_type=task_type,
            name=name,
            description=description,
            parameters=parameters,
            dependencies=dependencies or set(),
            priority=priority,
            max_retries=max_retries,
            timeout=timeout
        )
        
        self.tasks.append(task)
        return self
    
    def add_llm_task(
        self,
        name: str,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]] = None,
        description: str = "",
        dependencies: Set[str] = None,
        priority: int = 1,
        timeout: float = 300.0
    ) -> 'TaskChain':
        """添加 LLM 任务"""
        parameters = {
            "messages": messages,
            "tools": tools or []
        }
        
        return self.add_task(
            task_type=TaskType.LLM_COMPLETION,
            name=name,
            description=description or f"LLM 对话: {name}",
            parameters=parameters,
            dependencies=dependencies,
            priority=priority,
            timeout=timeout
        )
    
    def add_knowledge_search(
        self,
        name: str,
        query: str,
        top_k: int = 3,
        description: str = "",
        dependencies: Set[str] = None,
        priority: int = 1
    ) -> 'TaskChain':
        """添加知识搜索任务"""
        parameters = {
            "query": query,
            "top_k": top_k
        }
        
        return self.add_task(
            task_type=TaskType.KNOWLEDGE_SEARCH,
            name=name,
            description=description or f"知识搜索: {query}",
            parameters=parameters,
            dependencies=dependencies,
            priority=priority
        )
    
    def add_web_scraping(
        self,
        name: str,
        url: str,
        description: str = "",
        dependencies: Set[str] = None,
        priority: int = 1,
        timeout: float = 60.0
    ) -> 'TaskChain':
        """添加网页抓取任务"""
        parameters = {
            "url": url
        }
        
        return self.add_task(
            task_type=TaskType.WEB_SCRAPING,
            name=name,
            description=description or f"网页抓取: {url}",
            parameters=parameters,
            dependencies=dependencies,
            priority=priority,
            timeout=timeout
        )
    
    def add_tool_execution(
        self,
        name: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        description: str = "",
        dependencies: Set[str] = None,
        priority: int = 1
    ) -> 'TaskChain':
        """添加工具执行任务"""
        parameters = {
            "tool_name": tool_name,
            "tool_args": tool_args
        }
        
        return self.add_task(
            task_type=TaskType.TOOL_EXECUTION,
            name=name,
            description=description or f"工具执行: {tool_name}",
            parameters=parameters,
            dependencies=dependencies,
            priority=priority
        )
    
    def add_parallel_tasks(
        self,
        name: str,
        sub_tasks: List[Dict[str, Any]],
        description: str = "",
        dependencies: Set[str] = None,
        priority: int = 1
    ) -> 'TaskChain':
        """添加并行执行任务"""
        parameters = {
            "tasks": sub_tasks
        }
        
        return self.add_task(
            task_type=TaskType.PARALLEL_EXECUTION,
            name=name,
            description=description or f"并行执行: {name}",
            parameters=parameters,
            dependencies=dependencies,
            priority=priority
        )
    
    def add_conditional_tasks(
        self,
        name: str,
        condition: str,
        true_tasks: List[Dict[str, Any]],
        false_tasks: List[Dict[str, Any]] = None,
        description: str = "",
        dependencies: Set[str] = None,
        priority: int = 1
    ) -> 'TaskChain':
        """添加条件执行任务"""
        parameters = {
            "condition": condition,
            "true_tasks": true_tasks,
            "false_tasks": false_tasks or []
        }
        
        return self.add_task(
            task_type=TaskType.CONDITIONAL_EXECUTION,
            name=name,
            description=description or f"条件执行: {name}",
            parameters=parameters,
            dependencies=dependencies,
            priority=priority
        )
    
    async def execute(self, orchestrator: TaskOrchestrator = None, 
                     execution_context: TaskExecutionContext = None) -> List[TaskResult]:
        """执行任务链"""
        if not orchestrator and not self.orchestrator:
            raise ValueError("需要 TaskOrchestrator 实例")
        
        orchestrator = orchestrator or self.orchestrator
        execution_context = execution_context or self.execution_context
        
        if not orchestrator:
            raise ValueError("TaskOrchestrator 未设置")
        
        logger.info(f"开始执行任务链，包含 {len(self.tasks)} 个任务")
        
        # 提交所有任务
        task_ids = await orchestrator.submit_tasks(self.tasks, execution_context)
        
        # 等待所有任务完成
        results = []
        for task_id in task_ids:
            # 轮询任务状态
            while True:
                status = await orchestrator.get_task_status(task_id)
                if status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                    result = await orchestrator.get_task_result(task_id)
                    results.append(result)
                    break
                
                await asyncio.sleep(0.5)
        
        logger.info(f"任务链执行完成，成功: {len([r for r in results if r.status == TaskStatus.COMPLETED])}, 失败: {len([r for r in results if r.status == TaskStatus.FAILED])}")
        
        return results
    
    async def execute_sync(self, orchestrator: TaskOrchestrator = None,
                          execution_context: TaskExecutionContext = None) -> List[TaskResult]:
        """同步执行任务链"""
        if not orchestrator and not self.orchestrator:
            raise ValueError("需要 TaskOrchestrator 实例")
        
        orchestrator = orchestrator or self.orchestrator
        execution_context = execution_context or self.execution_context
        
        if not orchestrator:
            raise ValueError("TaskOrchestrator 未设置")
        
        results = []
        for task in self.tasks:
            # 检查依赖关系
            if task.dependencies:
                # 等待依赖任务完成
                for dep_id in task.dependencies:
                    dep_result = None
                    while not dep_result:
                        dep_result = await orchestrator.get_task_result(dep_id)
                        if dep_result and dep_result.status != TaskStatus.COMPLETED:
                            raise Exception(f"依赖任务失败: {dep_id}")
                        await asyncio.sleep(0.1)
            
            # 执行任务
            result = await orchestrator.execute_task_sync(task, execution_context)
            results.append(result)
            
            # 如果任务失败，可以选择停止整个链
            if result.status == TaskStatus.FAILED:
                logger.error(f"任务链中断: {task.name} 失败")
                break
        
        return results


class TaskTemplates:
    """任务模板库 - 预定义常用任务模式"""
    
    @staticmethod
    def create_research_workflow(query: str, search_web: bool = True, 
                                search_knowledge: bool = True) -> TaskChain:
        """创建研究工作流"""
        chain = TaskChain()
        
        # 步骤1: 知识搜索
        if search_knowledge:
            chain.add_knowledge_search(
                name="knowledge_research",
                query=query,
                top_k=5,
                description="搜索相关知识库"
            )
        
        # 步骤2: 网络搜索（如果启用）
        if search_web:
            # 这里可以添加网络搜索逻辑
            pass
        
        # 步骤3: LLM 综合分析
        messages = [
            {
                "role": "system",
                "content": "你是一个研究助手，基于搜索结果提供综合分析。"
            },
            {
                "role": "user", 
                "content": f"请基于搜索结果分析: {query}"
            }
        ]
        
        chain.add_llm_task(
            name="comprehensive_analysis",
            messages=messages,
            description="综合分析与研究"
        )
        
        return chain
    
    @staticmethod
    def create_content_analysis_workflow(content: str, 
                                       analysis_type: str = "comprehensive") -> TaskChain:
        """创建内容分析工作流"""
        chain = TaskChain()
        
        # 步骤1: 内容预处理
        preprocessing_messages = [
            {
                "role": "system",
                "content": "你是一个内容分析专家，请对提供的文本进行预处理和分析。"
            },
            {
                "role": "user",
                "content": f"请分析以下内容的结构和关键信息:\n\n{content}"
            }
        ]
        
        chain.add_llm_task(
            name="content_preprocessing",
            messages=preprocessing_messages,
            description="内容预处理分析"
        )
        
        # 步骤2: 根据分析类型进行专项分析
        if analysis_type == "comprehensive":
            analysis_messages = [
                {
                    "role": "system",
                    "content": "基于预处理结果，提供全面的内容分析，包括主题、情感、关键点等。"
                },
                {
                    "role": "user",
                    "content": "请提供全面的内容分析报告"
                }
            ]
        elif analysis_type == "sentiment":
            analysis_messages = [
                {
                    "role": "system",
                    "content": "基于预处理结果，分析内容的情感倾向。"
                },
                {
                    "role": "user",
                    "content": "请分析内容的情感倾向"
                }
            ]
        else:
            analysis_messages = [
                {
                    "role": "system",
                    "content": f"基于预处理结果，进行{analysis_type}分析。"
                },
                {
                    "role": "user",
                    "content": f"请进行{analysis_type}分析"
                }
            ]
        
        chain.add_llm_task(
            name="specialized_analysis",
            messages=analysis_messages,
            description=f"专项分析: {analysis_type}",
            dependencies={chain.tasks[-1].task_id} if chain.tasks else None
        )
        
        return chain
    
    @staticmethod
    def create_multi_source_research_workflow(
        query: str,
        sources: List[str] = None
    ) -> TaskChain:
        """创建多源研究工作流"""
        chain = TaskChain()
        sources = sources or ["knowledge", "web", "llm"]
        
        # 并行搜索多个来源
        parallel_tasks = []
        
        if "knowledge" in sources:
            parallel_tasks.append({
                "task_type": TaskType.KNOWLEDGE_SEARCH,
                "name": "knowledge_search",
                "description": "搜索知识库",
                "parameters": {"query": query, "top_k": 3}
            })
        
        if "web" in sources:
            # 这里可以添加网页搜索任务
            pass
        
        # 执行并行搜索
        chain.add_parallel_tasks(
            name="multi_source_search",
            sub_tasks=parallel_tasks,
            description="多源并行搜索"
        )
        
        # 综合分析
        synthesis_messages = [
            {
                "role": "system",
                "content": "你是一个综合分析专家，请整合多源搜索结果提供全面分析。"
            },
            {
                "role": "user",
                "content": f"请基于多源搜索结果综合分析: {query}"
            }
        ]
        
        chain.add_llm_task(
            name="synthesis_analysis",
            messages=synthesis_messages,
            description="综合分析与总结",
            dependencies={chain.tasks[-1].task_id} if chain.tasks else None
        )
        
        return chain
    
    @staticmethod
    def create_conversation_learning_workflow(
        user_message: str,
        bot_response: str,
        conversation_id: int = None,
        user_id: int = None
    ) -> TaskChain:
        """创建对话学习工作流"""
        chain = TaskChain()
        
        # 步骤1: 从对话中提取知识
        extraction_messages = [
            {
                "role": "system",
                "content": "你是一个知识提取专家，请从对话中提取有价值的知识点。"
            },
            {
                "role": "user",
                "content": f"请从以下对话中提取知识:\n用户: {user_message}\n助手: {bot_response}"
            }
        ]
        
        chain.add_llm_task(
            name="knowledge_extraction",
            messages=extraction_messages,
            description="从对话中提取知识"
        )
        
        # 步骤2: 存储学习结果
        chain.add_task(
            task_type=TaskType.LEARNING,
            name="store_knowledge",
            description="存储提取的知识",
            parameters={
                "user_message": user_message,
                "bot_response": bot_response,
                "conversation_id": conversation_id,
                "user_id": user_id
            },
            dependencies={chain.tasks[-1].task_id} if chain.tasks else None
        )
        
        return chain


class SmartTaskBuilder:
    """智能任务构建器 - 自动优化和构建任务"""
    
    def __init__(self, orchestrator: TaskOrchestrator):
        self.orchestrator = orchestrator
    
    async def build_intelligent_response_task(
        self,
        user_query: str,
        conversation_history: List[Dict[str, str]] = None,
        system_context: str = None,
        enable_web_search: bool = True,
        enable_knowledge_search: bool = True,
        enable_learning: bool = True
    ) -> TaskChain:
        """构建智能响应任务"""
        chain = TaskChain(orchestrator=self.orchestrator)
        
        # 分析查询意图
        intent_analysis_messages = [
            {
                "role": "system",
                "content": "你是一个查询意图分析专家，请分析用户的查询意图。"
            },
            {
                "role": "user",
                "content": f"请分析以下查询的意图，并判断需要什么类型的信息:\n{user_query}"
            }
        ]
        
        chain.add_llm_task(
            name="intent_analysis",
            messages=intent_analysis_messages,
            description="分析用户查询意图"
        )
        
        # 根据意图构建相应的任务
        # 这里可以添加更复杂的逻辑判断
        
        if enable_knowledge_search:
            chain.add_knowledge_search(
                name="personal_knowledge_search",
                query=user_query,
                description="搜索个人知识库",
                dependencies={chain.tasks[-1].task_id} if chain.tasks else None
            )
        
        if enable_web_search:
            # 判断是否需要网页搜索
            web_keywords = ["最新", "新闻", "实时", "当前", "今天", "现在", "网页", "网站"]
            if any(keyword in user_query for keyword in web_keywords):
                # 这里可以添加网页搜索逻辑
                pass
        
        # 构建最终响应
        final_messages = [
            {
                "role": "system",
                "content": system_context or "你是一个智能助手，基于搜索结果为用户提供准确、有用的回答。"
            }
        ]
        
        # 添加对话历史
        if conversation_history:
            final_messages.extend(conversation_history)
        
        # 添加当前查询
        final_messages.append({
            "role": "user",
            "content": user_query
        })
        
        chain.add_llm_task(
            name="generate_response",
            messages=final_messages,
            description="生成最终响应",
            dependencies={chain.tasks[-1].task_id} if chain.tasks else None
        )
        
        # 学习对话（如果启用）
        if enable_learning:
            chain.add_task(
                task_type=TaskType.LEARNING,
                name="learn_from_conversation",
                description="从对话中学习",
                parameters={
                    "user_message": user_query,
                    "bot_response": "${generate_response.result}",  # 引用前面的结果
                },
                dependencies={chain.tasks[-1].task_id} if chain.tasks else None
            )
        
        return chain
    
    async def build_complex_analysis_task(
        self,
        analysis_goal: str,
        data_sources: List[str] = None,
        analysis_methods: List[str] = None
    ) -> TaskChain:
        """构建复杂分析任务"""
        chain = TaskChain(orchestrator=self.orchestrator)
        data_sources = data_sources or ["knowledge", "web"]
        analysis_methods = analysis_methods or ["comprehensive", "comparative"]
        
        # 步骤1: 数据收集
        data_collection_tasks = []
        
        if "knowledge" in data_sources:
            data_collection_tasks.append({
                "task_type": TaskType.KNOWLEDGE_SEARCH,
                "name": "collect_from_knowledge",
                "description": "从知识库收集数据",
                "parameters": {"query": analysis_goal, "top_k": 5}
            })
        
        if "web" in data_sources:
            # 这里可以添加网页数据收集任务
            pass
        
        chain.add_parallel_tasks(
            name="data_collection",
            sub_tasks=data_collection_tasks,
            description="并行数据收集"
        )
        
        # 步骤2: 数据预处理
        preprocessing_messages = [
            {
                "role": "system",
                "content": "你是一个数据预处理专家，请对收集的数据进行清洗和预处理。"
            },
            {
                "role": "user",
                "content": f"请对收集的数据进行预处理，目标: {analysis_goal}"
            }
        ]
        
        chain.add_llm_task(
            name="data_preprocessing",
            messages=preprocessing_messages,
            description="数据预处理",
            dependencies={chain.tasks[-1].task_id} if chain.tasks else None
        )
        
        # 步骤3: 多方法分析
        analysis_tasks = []
        
        for method in analysis_methods:
            if method == "comprehensive":
                analysis_messages = [
                    {
                        "role": "system",
                        "content": "你是一个综合分析专家，请提供全面的分析。"
                    },
                    {
                        "role": "user",
                        "content": f"请对预处理后的数据进行综合分析，目标: {analysis_goal}"
                    }
                ]
            elif method == "comparative":
                analysis_messages = [
                    {
                        "role": "system",
                        "content": "你是一个比较分析专家，请进行对比分析。"
                    },
                    {
                        "role": "user",
                        "content": f"请对预处理后的数据进行比较分析，目标: {analysis_goal}"
                    }
                ]
            else:
                continue
            
            analysis_tasks.append({
                "task_type": TaskType.LLM_COMPLETION,
                "name": f"{method}_analysis",
                "description": f"{method}分析",
                "parameters": {"messages": analysis_messages}
            })
        
        chain.add_parallel_tasks(
            name="multi_method_analysis",
            sub_tasks=analysis_tasks,
            description="多方法并行分析",
            dependencies={chain.tasks[-1].task_id} if chain.tasks else None
        )
        
        # 步骤4: 结果综合
        synthesis_messages = [
            {
                "role": "system",
                "content": "你是一个分析结果综合专家，请整合多方法分析结果。"
            },
            {
                "role": "user",
                "content": f"请综合各种分析方法的结果，提供最终分析报告，目标: {analysis_goal}"
            }
        ]
        
        chain.add_llm_task(
            name="result_synthesis",
            messages=synthesis_messages,
            description="分析结果综合",
            dependencies={chain.tasks[-1].task_id} if chain.tasks else None
        )
        
        return chain