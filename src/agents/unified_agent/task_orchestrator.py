"""
任务编排器 - 统一 Agent 的任务编排与执行能力
- 支持复杂多步骤任务的自动规划与执行
- 任务依赖管理与并行执行
- 错误处理与重试机制
- 任务状态跟踪与恢复
- 动态任务生成与调整
"""

import asyncio
import json
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import traceback

from loguru import logger


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"
    TIMEOUT = "timeout"


class TaskType(Enum):
    """任务类型枚举"""
    LLM_COMPLETION = "llm_completion"
    KNOWLEDGE_SEARCH = "knowledge_search"
    WEB_SCRAPING = "web_scraping"
    TOOL_EXECUTION = "tool_execution"
    IMAGE_ANALYSIS = "image_analysis"
    LEARNING = "learning"
    RAG_QUERY = "rag_query"
    MCP_TOOL = "mcp_tool"
    CUSTOM_FUNCTION = "custom_function"
    PARALLEL_EXECUTION = "parallel_execution"
    SEQUENTIAL_EXECUTION = "sequential_execution"
    CONDITIONAL_EXECUTION = "conditional_execution"


@dataclass
class TaskResult:
    """任务执行结果"""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    """任务定义"""
    task_id: str
    task_type: TaskType
    name: str
    description: str
    parameters: Dict[str, Any]
    dependencies: Set[str] = field(default_factory=set)
    priority: int = 1
    max_retries: int = 3
    timeout: float = 300.0  # 5分钟默认超时
    retry_delay: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.task_id:
            self.task_id = str(uuid.uuid4())


@dataclass
class TaskExecutionContext:
    """任务执行上下文"""
    execution_id: str
    agent_service: Any  # UnifiedAgentService
    vector_service: Any
    mcp_service: Any
    database_service: Any
    conversation_id: Optional[int] = None
    user_id: Optional[int] = None
    global_context: Dict[str, Any] = field(default_factory=dict)
    task_results: Dict[str, TaskResult] = field(default_factory=dict)


class TaskExecutor:
    """任务执行器 - 负责具体任务的执行"""
    
    def __init__(self, context: TaskExecutionContext):
        self.context = context
        self._executor_pool = ThreadPoolExecutor(max_workers=10)
    
    async def execute_task(self, task: Task) -> TaskResult:
        """执行单个任务"""
        start_time = datetime.now()
        task_result = TaskResult(
            task_id=task.task_id,
            status=TaskStatus.RUNNING,
            start_time=start_time
        )
        
        try:
            logger.info(f"开始执行任务: {task.name} ({task.task_type.value})")
            
            # 根据任务类型执行相应的操作
            result = await self._execute_task_by_type(task)
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            task_result.status = TaskStatus.COMPLETED
            task_result.result = result
            task_result.end_time = end_time
            task_result.execution_time = execution_time
            
            logger.info(f"任务执行成功: {task.name} (耗时: {execution_time:.2f}s)")
            
        except asyncio.TimeoutError:
            task_result.status = TaskStatus.TIMEOUT
            task_result.error = f"任务超时 ({task.timeout}s)"
            logger.error(f"任务超时: {task.name}")
            
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            task_result.status = TaskStatus.FAILED
            task_result.error = f"{type(e).__name__}: {str(e)}"
            task_result.end_time = end_time
            task_result.execution_time = execution_time
            
            logger.error(f"任务执行失败: {task.name} - {task_result.error}")
            logger.error(traceback.format_exc())
        
        return task_result
    
    async def _execute_task_by_type(self, task: Task) -> Any:
        """根据任务类型执行具体逻辑"""
        task_type = task.task_type
        params = task.parameters
        
        if task_type == TaskType.LLM_COMPLETION:
            return await self._execute_llm_completion(params)
        
        elif task_type == TaskType.KNOWLEDGE_SEARCH:
            return await self._execute_knowledge_search(params)
        
        elif task_type == TaskType.WEB_SCRAPING:
            return await self._execute_web_scraping(params)
        
        elif task_type == TaskType.TOOL_EXECUTION:
            return await self._execute_tool_execution(params)
        
        elif task_type == TaskType.IMAGE_ANALYSIS:
            return await self._execute_image_analysis(params)
        
        elif task_type == TaskType.LEARNING:
            return await self._execute_learning(params)
        
        elif task_type == TaskType.RAG_QUERY:
            return await self._execute_rag_query(params)
        
        elif task_type == TaskType.MCP_TOOL:
            return await self._execute_mcp_tool(params)
        
        elif task_type == TaskType.CUSTOM_FUNCTION:
            return await self._execute_custom_function(params)
        
        elif task_type == TaskType.PARALLEL_EXECUTION:
            return await self._execute_parallel_execution(params)
        
        elif task_type == TaskType.SEQUENTIAL_EXECUTION:
            return await self._execute_sequential_execution(params)
        
        elif task_type == TaskType.CONDITIONAL_EXECUTION:
            return await self._execute_conditional_execution(params)
        
        else:
            raise ValueError(f"不支持的任务类型: {task_type}")
    
    async def _execute_llm_completion(self, params: Dict[str, Any]) -> str:
        """执行 LLM 完成"""
        messages = params.get("messages", [])
        tools = params.get("tools", [])
        
        if not self.context.agent_service:
            raise ValueError("Agent 服务未初始化")
        
        result = await self.context.agent_service.chat_completion(messages, tools)
        return result
    
    async def _execute_knowledge_search(self, params: Dict[str, Any]) -> str:
        """执行知识搜索"""
        query = params.get("query", "")
        top_k = params.get("top_k", 3)
        
        if not self.context.vector_service:
            raise ValueError("向量服务未初始化")
        
        results = await self.context.vector_service.search_knowledge(query, top_k=top_k)
        return results
    
    async def _execute_web_scraping(self, params: Dict[str, Any]) -> str:
        """执行网页抓取"""
        url = params.get("url", "")
        
        if not self.context.agent_service:
            raise ValueError("Agent 服务未初始化")
        
        # 复用 unified_agent_service 中的网页抓取逻辑
        result = await self.context.agent_service._scrape_webpage(url)
        return result
    
    async def _execute_tool_execution(self, params: Dict[str, Any]) -> str:
        """执行工具调用"""
        tool_name = params.get("tool_name", "")
        tool_args = params.get("tool_args", {})
        
        if not self.context.agent_service:
            raise ValueError("Agent 服务未初始化")
        
        result = await self.context.agent_service._handle_mcp_tool(tool_name, tool_args)
        return result
    
    async def _execute_image_analysis(self, params: Dict[str, Any]) -> str:
        """执行图片分析"""
        image_data = params.get("image_data", b"")
        prompt = params.get("prompt", "请描述这张图片")
        
        if not self.context.agent_service:
            raise ValueError("Agent 服务未初始化")
        
        result = await self.context.agent_service.analyze_image(image_data, prompt)
        return result
    
    async def _execute_learning(self, params: Dict[str, Any]) -> bool:
        """执行学习任务"""
        user_message = params.get("user_message", "")
        bot_response = params.get("bot_response", "")
        conversation_id = params.get("conversation_id", self.context.conversation_id)
        user_id = params.get("user_id", self.context.user_id)
        
        if not self.context.agent_service:
            raise ValueError("Agent 服务未初始化")
        
        await self.context.agent_service.learn_from_conversation(
            user_message, bot_response, conversation_id, user_id
        )
        return True
    
    async def _execute_rag_query(self, params: Dict[str, Any]) -> str:
        """执行 RAG 查询"""
        query = params.get("query", "")
        top_k = params.get("top_k", 3)
        
        if not self.context.agent_service:
            raise ValueError("Agent 服务未初始化")
        
        # 复用 unified_agent_service 中的 RAG 逻辑
        result = await self.context.agent_service._retrieve_knowledge(query, top_k=top_k)
        return result
    
    async def _execute_mcp_tool(self, params: Dict[str, Any]) -> str:
        """执行 MCP 工具"""
        tool_name = params.get("tool_name", "")
        tool_args = params.get("tool_args", {})
        
        if not self.context.mcp_service:
            raise ValueError("MCP 服务未初始化")
        
        result = await self.context.mcp_service.call_tool(tool_name, tool_args)
        return result
    
    async def _execute_custom_function(self, params: Dict[str, Any]) -> Any:
        """执行自定义函数"""
        function_name = params.get("function_name", "")
        function_args = params.get("function_args", {})
        
        # 从全局上下文中获取函数
        custom_function = self.context.global_context.get(function_name)
        if not custom_function:
            raise ValueError(f"未找到自定义函数: {function_name}")
        
        if asyncio.iscoroutinefunction(custom_function):
            result = await custom_function(**function_args)
        else:
            # 对于同步函数，在线程池中执行
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor_pool, custom_function, **function_args
            )
        
        return result
    
    async def _execute_parallel_execution(self, params: Dict[str, Any]) -> List[Any]:
        """执行并行任务"""
        sub_tasks = params.get("tasks", [])
        
        if not sub_tasks:
            return []
        
        # 创建子任务执行器
        sub_executor = TaskExecutor(self.context)
        
        # 并行执行所有子任务
        tasks = []
        for sub_task_data in sub_tasks:
            sub_task = Task(**sub_task_data)
            task_coro = sub_executor.execute_task(sub_task)
            tasks.append(task_coro)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
    
    async def _execute_sequential_execution(self, params: Dict[str, Any]) -> Any:
        """执行顺序任务"""
        sub_tasks = params.get("tasks", [])
        
        if not sub_tasks:
            return None
        
        # 创建子任务执行器
        sub_executor = TaskExecutor(self.context)
        
        # 顺序执行所有子任务
        final_result = None
        for sub_task_data in sub_tasks:
            sub_task = Task(**sub_task_data)
            result = await sub_executor.execute_task(sub_task)
            
            if result.status == TaskStatus.FAILED:
                raise Exception(f"子任务失败: {sub_task.name} - {result.error}")
            
            final_result = result.result
        
        return final_result
    
    async def _execute_conditional_execution(self, params: Dict[str, Any]) -> Any:
        """执行条件任务"""
        condition = params.get("condition", "")
        true_tasks = params.get("true_tasks", [])
        false_tasks = params.get("false_tasks", [])
        
        # 评估条件
        condition_result = await self._evaluate_condition(condition)
        
        # 根据条件结果执行相应的任务
        if condition_result:
            tasks_to_execute = true_tasks
        else:
            tasks_to_execute = false_tasks
        
        if not tasks_to_execute:
            return None
        
        # 创建子任务执行器
        sub_executor = TaskExecutor(self.context)
        
        # 执行任务
        results = []
        for task_data in tasks_to_execute:
            task = Task(**task_data)
            result = await sub_executor.execute_task(task)
            results.append(result)
        
        return results
    
    async def _evaluate_condition(self, condition: str) -> bool:
        """评估条件表达式"""
        # 这里可以实现简单的条件评估逻辑
        # 例如：检查某个变量的值，或者执行简单的逻辑判断
        try:
            # 从全局上下文中获取变量值
            context = self.context.global_context
            
            # 简单的条件评估（可以扩展更复杂的逻辑）
            if condition.startswith("${") and condition.endswith("}"):
                # 变量引用，如: ${variable_name}
                var_name = condition[2:-1]
                return bool(context.get(var_name, False))
            else:
                # 简单的布尔值转换
                return bool(condition)
        except Exception as e:
            logger.warning(f"条件评估失败: {condition} - {e}")
            return False
    
    def cleanup(self):
        """清理资源"""
        if self._executor_pool:
            self._executor_pool.shutdown(wait=True)


class TaskOrchestrator:
    """任务编排器 - 负责任务调度与管理"""
    
    def __init__(self, agent_service=None, vector_service=None, 
                 mcp_service=None, database_service=None):
        self.agent_service = agent_service
        self.vector_service = vector_service
        self.mcp_service = mcp_service
        self.database_service = database_service
        
        # 任务队列
        self.task_queue: asyncio.Queue[Task] = asyncio.Queue()
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.completed_tasks: Dict[str, TaskResult] = {}
        self.failed_tasks: Dict[str, TaskResult] = {}
        
        # 执行状态
        self.is_running = False
        self.max_concurrent_tasks = 5
        self.semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        
        # 执行上下文缓存
        self._context_cache: Dict[str, TaskExecutionContext] = {}
    
    async def initialize(self) -> bool:
        """初始化任务编排器"""
        try:
            logger.info("正在初始化任务编排器...")
            self.is_running = True
            logger.info("任务编排器初始化完成")
            return True
        except Exception as e:
            logger.error(f"任务编排器初始化失败: {e}")
            return False
    
    async def create_task(
        self,
        task_type: TaskType,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        dependencies: Set[str] = None,
        priority: int = 1,
        max_retries: int = 3,
        timeout: float = 300.0
    ) -> Task:
        """创建新任务"""
        task = Task(
            task_id=str(uuid.uuid4()),
            task_type=task_type,
            name=name,
            description=description,
            parameters=parameters,
            dependencies=dependencies or set(),
            priority=priority,
            max_retries=max_retries,
            timeout=timeout
        )
        
        logger.info(f"创建任务: {task.name} ({task.task_id})")
        return task
    
    async def submit_task(self, task: Task, execution_context: TaskExecutionContext = None) -> str:
        """提交任务到执行队列"""
        try:
            # 创建执行上下文（如果未提供）
            if not execution_context:
                execution_context = await self._create_execution_context()
            
            # 缓存执行上下文
            self._context_cache[task.task_id] = execution_context
            
            # 将任务添加到队列
            await self.task_queue.put(task)
            
            logger.info(f"任务已提交: {task.name} ({task.task_id})")
            return task.task_id
            
        except Exception as e:
            logger.error(f"提交任务失败: {task.name} - {e}")
            raise
    
    async def submit_tasks(self, tasks: List[Task], execution_context: TaskExecutionContext = None) -> List[str]:
        """批量提交任务"""
        task_ids = []
        
        for task in tasks:
            task_id = await self.submit_task(task, execution_context)
            task_ids.append(task_id)
        
        return task_ids
    
    async def execute_task_sync(self, task: Task, execution_context: TaskExecutionContext = None) -> TaskResult:
        """同步执行单个任务"""
        try:
            # 创建执行上下文（如果未提供）
            if not execution_context:
                execution_context = await self._create_execution_context()
            
            # 执行任务
            executor = TaskExecutor(execution_context)
            result = await executor.execute_task(task)
            
            # 记录结果
            if result.status == TaskStatus.COMPLETED:
                self.completed_tasks[task.task_id] = result
            else:
                self.failed_tasks[task.task_id] = result
            
            return result
            
        except Exception as e:
            logger.error(f"同步执行任务失败: {task.name} - {e}")
            raise
    
    async def start_orchestrator(self):
        """启动任务编排器"""
        if self.is_running:
            logger.warning("任务编排器已在运行中")
            return
        
        self.is_running = True
        logger.info("任务编排器已启动")
        
        # 启动任务处理循环
        asyncio.create_task(self._task_processing_loop())
    
    async def stop_orchestrator(self):
        """停止任务编排器"""
        self.is_running = False
        logger.info("任务编排器已停止")
    
    async def _task_processing_loop(self):
        """任务处理主循环"""
        while self.is_running:
            try:
                # 从队列获取任务
                task = await self.task_queue.get()
                
                # 检查依赖关系
                if not await self._check_dependencies(task):
                    # 依赖未满足，重新放入队列
                    await asyncio.sleep(1)
                    await self.task_queue.put(task)
                    continue
                
                # 启动任务执行
                asyncio.create_task(self._execute_task_with_retry(task))
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"任务处理循环错误: {e}")
                await asyncio.sleep(1)
    
    async def _execute_task_with_retry(self, task: Task):
        """执行任务（支持重试）"""
        async with self.semaphore:
            retry_count = 0
            
            while retry_count <= task.max_retries:
                try:
                    # 获取执行上下文
                    execution_context = self._context_cache.get(task.task_id)
                    if not execution_context:
                        execution_context = await self._create_execution_context()
                    
                    # 执行任务
                    executor = TaskExecutor(execution_context)
                    result = await executor.execute_task(task)
                    
                    # 处理结果
                    if result.status == TaskStatus.COMPLETED:
                        self.completed_tasks[task.task_id] = result
                        logger.info(f"任务执行成功: {task.name}")
                        break
                    
                    elif result.status == TaskStatus.FAILED:
                        if retry_count < task.max_retries:
                            retry_count += 1
                            result.retry_count = retry_count
                            logger.warning(f"任务执行失败，准备重试: {task.name} (第{retry_count}次)")
                            await asyncio.sleep(task.retry_delay * retry_count)
                        else:
                            self.failed_tasks[task.task_id] = result
                            logger.error(f"任务执行失败（达到最大重试次数）: {task.name}")
                            break
                    
                    else:
                        # 其他状态（如超时、取消等）
                        self.failed_tasks[task.task_id] = result
                        logger.error(f"任务执行异常: {task.name} - {result.status.value}")
                        break
                
                except Exception as e:
                    retry_count += 1
                    logger.error(f"任务执行异常: {task.name} - {e} (第{retry_count}次)")
                    
                    if retry_count > task.max_retries:
                        # 创建失败结果
                        result = TaskResult(
                            task_id=task.task_id,
                            status=TaskStatus.FAILED,
                            error=f"{type(e).__name__}: {str(e)}",
                            retry_count=retry_count
                        )
                        self.failed_tasks[task.task_id] = result
                        break
                    
                    await asyncio.sleep(task.retry_delay * retry_count)
    
    async def _check_dependencies(self, task: Task) -> bool:
        """检查任务依赖关系"""
        if not task.dependencies:
            return True
        
        # 检查所有依赖任务是否已完成
        for dep_task_id in task.dependencies:
            if dep_task_id not in self.completed_tasks:
                return False
        
        return True
    
    async def _create_execution_context(self, conversation_id: int = None, user_id: int = None) -> TaskExecutionContext:
        """创建任务执行上下文"""
        execution_id = str(uuid.uuid4())
        
        return TaskExecutionContext(
            execution_id=execution_id,
            agent_service=self.agent_service,
            vector_service=self.vector_service,
            mcp_service=self.mcp_service,
            database_service=self.database_service,
            conversation_id=conversation_id,
            user_id=user_id,
            global_context={},
            task_results={}
        )
    
    async def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """获取任务状态"""
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id].status
        elif task_id in self.failed_tasks:
            return self.failed_tasks[task_id].status
        elif task_id in self.running_tasks:
            return TaskStatus.RUNNING
        else:
            return None
    
    async def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """获取任务结果"""
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
        elif task_id in self.failed_tasks:
            return self.failed_tasks[task_id]
        else:
            return None
    
    async def get_all_task_statuses(self) -> Dict[str, TaskStatus]:
        """获取所有任务状态"""
        statuses = {}
        
        # 已完成任务
        for task_id, result in self.completed_tasks.items():
            statuses[task_id] = result.status
        
        # 失败任务
        for task_id, result in self.failed_tasks.items():
            statuses[task_id] = result.status
        
        # 运行中任务
        for task_id in self.running_tasks:
            statuses[task_id] = TaskStatus.RUNNING
        
        return statuses
    
    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        if task_id in self.running_tasks:
            task = self.running_tasks[task_id]
            task.cancel()
            del self.running_tasks[task_id]
            
            # 记录取消结果
            result = TaskResult(
                task_id=task_id,
                status=TaskStatus.CANCELLED,
                error="任务被取消"
            )
            self.failed_tasks[task_id] = result
            
            logger.info(f"任务已取消: {task_id}")
            return True
        
        return False
    
    async def cleanup(self):
        """清理资源"""
        try:
            # 停止编排器
            await self.stop_orchestrator()
            
            # 清理上下文缓存
            self._context_cache.clear()
            
            # 清理执行器池
            for task in self.running_tasks.values():
                if not task.done():
                    task.cancel()
            
            self.running_tasks.clear()
            self.completed_tasks.clear()
            self.failed_tasks.clear()
            
            logger.info("任务编排器资源清理完成")
            
        except Exception as e:
            logger.error(f"任务编排器清理失败: {e}")
    
    def __del__(self):
        """析构函数"""
        try:
            if self.is_running:
                asyncio.create_task(self.cleanup())
        except Exception:
            pass