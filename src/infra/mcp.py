"""
MCP (Model Context Protocol) 服务模块
"""

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    MCP_AVAILABLE = True
except ImportError:
    logger.warning("MCP库未安装，MCP功能将不可用")
    MCP_AVAILABLE = False


class MCPServerConfig:
    """MCP服务器配置"""

    def __init__(
        self,
        name: str,
        command: str,
        args: List[str] = None,
        env: Dict[str, str] = None,
    ):
        self.name = name
        self.command = command
        self.args = args or []
        self.env = env or {}


class MCPService:
    """MCP服务管理类"""

    def __init__(self):
        self.sessions: Dict[str, ClientSession] = {}
        self.servers: Dict[str, MCPServerConfig] = {}
        self.tools: Dict[str, Dict[str, Any]] = {}

    async def add_server(self, server_config: MCPServerConfig) -> bool:
        """添加MCP服务器"""
        if not MCP_AVAILABLE:
            logger.error("MCP库不可用，无法连接服务器")
            return False

        try:
            logger.info(f"正在连接MCP服务器: {server_config.name}")

            # 创建服务器参数
            server_params = StdioServerParameters(
                command=server_config.command,
                args=server_config.args,
                env=server_config.env,
            )

            # 使用超时和更简单的连接方式
            try:
                # 临时连接以获取工具信息
                async with asyncio.timeout(10):  # 10秒超时
                    async with stdio_client(server_params) as (read, write):
                        async with ClientSession(read, write) as session:
                            # 初始化会话
                            await session.initialize()

                            # 获取服务器信息（某些版本可能不支持此方法）
                            try:
                                server_info = await session.get_server_info()
                                logger.info(
                                    f"MCP服务器 {server_config.name} 连接成功: {server_info}"
                                )
                            except AttributeError:
                                logger.info(
                                    f"MCP服务器 {server_config.name} 连接成功（服务器信息不可用）"
                                )

                            # 列出可用工具
                            tools_result = await session.list_tools()
                            if tools_result.tools:
                                self.tools[server_config.name] = {}
                                for tool in tools_result.tools:
                                    self.tools[server_config.name][tool.name] = {
                                        "description": tool.description,
                                        "schema": tool.inputSchema,
                                        "server": server_config.name,
                                    }
                                    logger.info(
                                        f"发现工具: {tool.name} - {tool.description}"
                                    )
                            else:
                                logger.warning(
                                    f"MCP服务器 {server_config.name} 未提供任何工具"
                                )

                # 保存配置（不保存会话，每次使用时重新连接）
                self.servers[server_config.name] = server_config

                return True

            except asyncio.TimeoutError:
                logger.error(f"连接MCP服务器 {server_config.name} 超时")
                return False

        except Exception as e:
            logger.error(f"连接MCP服务器 {server_config.name} 失败: {e}")
            return False

    async def remove_server(self, server_name: str) -> bool:
        """移除MCP服务器"""
        try:
            if server_name in self.sessions:
                # 关闭会话
                session = self.sessions[server_name]
                await session.close()

                # 清理
                del self.sessions[server_name]
                del self.servers[server_name]
                if server_name in self.tools:
                    del self.tools[server_name]

                logger.info(f"MCP服务器 {server_name} 已断开连接")
                return True
            else:
                logger.warning(f"MCP服务器 {server_name} 不存在")
                return False

        except Exception as e:
            logger.error(f"断开MCP服务器 {server_name} 失败: {e}")
            return False

    async def call_tool(
        self, tool_name: str, arguments: Dict[str, Any], server_name: str = None
    ) -> Optional[str]:
        """调用MCP工具"""
        if not MCP_AVAILABLE:
            return "MCP库不可用，无法调用工具"

        try:
            # 查找工具所属服务器
            target_server = None
            if server_name:
                if server_name in self.tools and tool_name in self.tools[server_name]:
                    target_server = server_name
            else:
                # 搜索所有服务器
                for srv_name, tools in self.tools.items():
                    if tool_name in tools:
                        target_server = srv_name
                        break

            if not target_server:
                logger.error(f"工具 {tool_name} 未找到")
                return None

            if target_server not in self.servers:
                logger.error(f"MCP服务器 {target_server} 配置不存在")
                return None

            server_config = self.servers[target_server]

            # 创建服务器参数
            server_params = StdioServerParameters(
                command=server_config.command,
                args=server_config.args,
                env=server_config.env,
            )

            # 创建临时连接调用工具，添加超时
            try:
                async with asyncio.timeout(30):  # 30秒超时
                    async with stdio_client(server_params) as (read, write):
                        async with ClientSession(read, write) as session:
                            # 初始化会话
                            await session.initialize()

                            # 调用工具
                            logger.info(f"调用工具: {tool_name} 参数: {arguments}")
                            result = await session.call_tool(tool_name, arguments)

                            if result.content:
                                # 处理结果内容
                                content_parts = []
                                for content in result.content:
                                    if hasattr(content, "text"):
                                        content_parts.append(content.text)
                                    elif hasattr(content, "data"):
                                        content_parts.append(str(content.data))
                                    else:
                                        content_parts.append(str(content))

                                result_text = "\n".join(content_parts)
                                logger.info(f"工具 {tool_name} 调用成功")
                                return result_text
                            else:
                                logger.warning(f"工具 {tool_name} 返回空结果")
                                return "工具执行完成，但没有返回内容"

            except asyncio.TimeoutError:
                logger.error(f"调用工具 {tool_name} 超时")
                return f"工具调用超时: {tool_name}"

        except Exception as e:
            logger.error(f"调用工具 {tool_name} 失败: {e}")
            return f"工具调用失败: {str(e)}"

    def get_available_tools(self) -> Dict[str, Dict[str, Any]]:
        """获取所有可用工具"""
        all_tools = {}
        for server_name, tools in self.tools.items():
            for tool_name, tool_info in tools.items():
                all_tools[tool_name] = tool_info
        return all_tools

    def get_tools_for_llm(self) -> List[Dict[str, Any]]:
        """获取适用于LLM function calling的工具定义"""
        tools_for_llm = []

        for server_name, tools in self.tools.items():
            for tool_name, tool_info in tools.items():
                # 转换为OpenAI function calling格式
                function_def = {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": tool_info.get("description", ""),
                        "parameters": tool_info.get("schema", {}),
                    },
                }
                tools_for_llm.append(function_def)

        return tools_for_llm

    def get_server_status(self) -> Dict[str, Dict[str, Any]]:
        """获取所有服务器状态"""
        status = {}
        for server_name, server_config in self.servers.items():
            status[server_name] = {
                "name": server_name,
                "command": server_config.command,
                "args": server_config.args,
                "connected": server_name in self.tools,  # 如果有工具则认为可连接
                "tools_count": len(self.tools.get(server_name, {})),
                "tools": list(self.tools.get(server_name, {}).keys()),
            }
        return status

    async def initialize_default_servers(self, mcp_configs: List[Dict[str, Any]]):
        """初始化默认MCP服务器"""
        for config in mcp_configs:
            try:
                server_config = MCPServerConfig(
                    name=config.get("name", "unknown"),
                    command=config.get("command", ""),
                    args=config.get("args", []),
                    env=config.get("env", {}),
                )

                success = await self.add_server(server_config)
                if success:
                    logger.info(f"MCP服务器 {server_config.name} 初始化成功")
                else:
                    logger.error(f"MCP服务器 {server_config.name} 初始化失败")

            except Exception as e:
                logger.error(f"初始化MCP服务器失败: {e}")

    async def close_all(self):
        """关闭所有MCP连接"""
        # 清理所有数据（不需要关闭会话，因为每次都是临时连接）
        self.sessions.clear()
        self.servers.clear()
        self.tools.clear()

        logger.info("所有MCP服务器连接已关闭")
