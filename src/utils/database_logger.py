"""
数据库记录辅助工具
用于在各个模块中方便地记录数据库日志
"""

import asyncio
import time
from typing import Any, Dict, Optional

from loguru import logger


class DatabaseLogger:
    """数据库记录辅助类"""

    def __init__(self, database_service=None):
        self.database_service = database_service

    def log_user_action(
        self, user_id: int, action: str, details: Dict[str, Any] = None
    ):
        """记录用户行为（异步执行，不阻塞主流程）"""
        if not self.database_service:
            return

        try:
            # 创建异步任务，不等待结果
            asyncio.create_task(self._log_user_action_async(user_id, action, details))
        except Exception as e:
            logger.warning(f"创建用户行为记录任务失败: {e}")

    async def _log_user_action_async(
        self, user_id: int, action: str, details: Dict[str, Any] = None
    ):
        """异步记录用户行为"""
        try:
            # 这里可以扩展为记录用户行为到专门的表
            logger.info(
                f"用户行为记录: user_id={user_id}, action={action}, details={details}"
            )
        except Exception as e:
            logger.warning(f"记录用户行为失败: {e}")

    def log_system_event(
        self, event_type: str, description: str, metadata: Dict[str, Any] = None
    ):
        """记录系统事件"""
        if not self.database_service:
            return

        try:
            asyncio.create_task(
                self._log_system_event_async(event_type, description, metadata)
            )
        except Exception as e:
            logger.warning(f"创建系统事件记录任务失败: {e}")

    async def _log_system_event_async(
        self, event_type: str, description: str, metadata: Dict[str, Any] = None
    ):
        """异步记录系统事件"""
        try:
            # 这里可以扩展为记录系统事件到专门的表
            logger.info(
                f"系统事件记录: type={event_type}, description={description}, metadata={metadata}"
            )
        except Exception as e:
            logger.warning(f"记录系统事件失败: {e}")

    def log_error(
        self, error_type: str, error_message: str, context: Dict[str, Any] = None
    ):
        """记录错误信息"""
        if not self.database_service:
            return

        try:
            asyncio.create_task(
                self._log_error_async(error_type, error_message, context)
            )
        except Exception as e:
            logger.warning(f"创建错误记录任务失败: {e}")

    async def _log_error_async(
        self, error_type: str, error_message: str, context: Dict[str, Any] = None
    ):
        """异步记录错误信息"""
        try:
            # 这里可以扩展为记录错误到专门的表
            logger.error(
                f"错误记录: type={error_type}, message={error_message}, context={context}"
            )
        except Exception as e:
            logger.warning(f"记录错误失败: {e}")


# 全局数据库记录器实例（在bot初始化时设置）
db_logger: Optional[DatabaseLogger] = None


def init_database_logger(database_service):
    """初始化全局数据库记录器"""
    global db_logger
    db_logger = DatabaseLogger(database_service)
    logger.info("数据库记录器初始化完成")


def log_user_action(user_id: int, action: str, details: Dict[str, Any] = None):
    """全局用户行为记录函数"""
    if db_logger:
        db_logger.log_user_action(user_id, action, details)


def log_system_event(
    event_type: str, description: str, metadata: Dict[str, Any] = None
):
    """全局系统事件记录函数"""
    if db_logger:
        db_logger.log_system_event(event_type, description, metadata)


def log_error(error_type: str, error_message: str, context: Dict[str, Any] = None):
    """全局错误记录函数"""
    if db_logger:
        db_logger.log_error(error_type, error_message, context)
