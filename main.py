#!/usr/bin/env python3
"""
AL1S-Bot 主程序入口
"""
import signal
import sys
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from loguru import logger
from src.bot import AL1SBot


def signal_handler(signum, frame):
    """信号处理器"""
    logger.info(f"收到信号 {signum}，正在关闭机器人...")
    sys.exit(0)


def main():
    """主函数"""
    try:
        # 配置日志
        logger.add(
            "logs/bot.log",
            rotation="1 day",
            retention="7 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
        )
        
        logger.info("正在启动 AL1S-Bot...")
        
        # 设置信号处理器
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # 创建机器人实例
        bot = AL1SBot()
        
        # 启动机器人
        bot.start()
        
    except KeyboardInterrupt:
        logger.info("收到键盘中断信号")
    except Exception as e:
        logger.error(f"机器人运行失败: {e}")
        raise


if __name__ == "__main__":
    try:
        # 创建logs目录
        Path("logs").mkdir(exist_ok=True)
        
        # 运行主程序
        main()
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except Exception as e:
        logger.error(f"程序启动失败: {e}")
        sys.exit(1)

