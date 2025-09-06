"""
Ascii2D图片搜索服务模块
使用PicImageSearch库进行图片搜索
"""

from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from loguru import logger
from PicImageSearch import Ascii2D, Network
from PicImageSearch.model import Ascii2DResponse

from ..config import config
from ..models import ImageSearchResult


class Ascii2DService:
    """Ascii2D图片搜索服务类"""

    def __init__(self):
        """初始化Ascii2D服务"""
        self.base_url = config.ascii2d.base_url
        self.bovw = config.ascii2d.bovw
        self.max_results = 10  # 硬编码默认值
        logger.info(
            f"Ascii2D服务初始化完成 - 基础URL: {self.base_url}, BOVW: {self.bovw}"
        )

    async def search_by_image_url(self, image_url: str) -> List[ImageSearchResult]:
        """通过图片URL搜索相似图片"""
        try:
            logger.info(f"开始搜索图片URL: {image_url}")

            # 使用PicImageSearch进行搜索
            async with Network(verify_ssl=True) as client:
                ascii2d = Ascii2D(base_url=self.base_url, bovw=self.bovw, client=client)

                # 执行搜索
                response: Ascii2DResponse = await ascii2d.search(url=image_url)

                if response and hasattr(response, "raw") and response.raw:
                    logger.info(f"搜索成功，找到 {len(response.raw)} 个结果")
                    # 转换搜索结果
                    results = self._convert_search_results_from_raw(
                        response.raw, "image_url"
                    )
                    logger.info(f"URL搜索完成，找到 {len(results)} 个结果")
                    return results
                else:
                    logger.warning("Ascii2D搜索未返回结果")
                    return []

        except Exception as e:
            logger.error(f"图片URL搜索失败: {e}")
            return []

    async def search_by_image_file(self, image_data: bytes) -> List[ImageSearchResult]:
        """通过图片文件搜索相似图片"""
        try:
            logger.info(f"开始搜索图片文件，大小: {len(image_data)} bytes")

            # 使用PicImageSearch进行搜索
            async with Network(verify_ssl=True) as client:
                ascii2d = Ascii2D(base_url=self.base_url, bovw=self.bovw, client=client)

                logger.info(f"使用基础URL: {self.base_url}, BOVW: {self.bovw}")

                # 执行搜索
                response: Ascii2DResponse = await ascii2d.search(file=image_data)

                if response and hasattr(response, "raw") and response.raw:
                    logger.info(f"搜索成功，找到 {len(response.raw)} 个结果")
                    # 转换搜索结果
                    results = self._convert_search_results_from_raw(
                        response.raw, "uploaded_image"
                    )
                    logger.info(f"文件搜索完成，找到 {len(results)} 个结果")
                    return results
                else:
                    logger.warning("Ascii2D搜索未返回结果")
                    return []

        except Exception as e:
            logger.error(f"图片文件搜索失败: {e}")
            return []

    def _convert_search_results_from_raw(
        self, raw_results: list, source_type: str
    ) -> List[ImageSearchResult]:
        """直接从原始搜索结果列表转换"""
        results = []

        try:
            # 限制结果数量
            limited_results = raw_results[: self.max_results]

            for i, result in enumerate(limited_results):
                try:
                    # 提取图片信息
                    img_url = self._extract_image_url(result)
                    if not img_url:
                        continue

                    # 提取标题
                    title = self._extract_title(result)

                    # 提取相似度（Ascii2D可能不提供相似度）
                    similarity = None

                    # 提取来源信息
                    source = self._extract_source(result)

                    # 创建结果对象
                    image_result = ImageSearchResult(
                        source=source,
                        url=img_url,
                        title=title,
                        similarity=similarity,
                        metadata={
                            "search_type": source_type,
                            "result_index": i + 1,
                            "author": getattr(result, "author", None),
                            "author_url": getattr(result, "author_url", None),
                            "hash": getattr(result, "hash", None),
                            "detail": getattr(result, "detail", None),
                            "thumbnail": getattr(result, "thumbnail", None),
                            "raw_result": str(result)[:200],  # 保存部分原始结果用于调试
                        },
                    )

                    results.append(image_result)

                except Exception as e:
                    logger.warning(f"转换第 {i+1} 个搜索结果失败: {e}")
                    continue

            logger.debug(f"成功转换 {len(results)} 个搜索结果")

        except Exception as e:
            logger.error(f"转换搜索结果失败: {e}")

        return results

    def _extract_image_url(self, result) -> Optional[str]:
        """从搜索结果中提取图片URL"""
        try:
            # 尝试多种可能的属性名
            url_attributes = ["url", "image_url", "src", "link", "href"]

            for attr in url_attributes:
                if hasattr(result, attr):
                    url = getattr(result, attr)
                    if url and isinstance(url, str):
                        return url

            # 如果是字典类型
            if isinstance(result, dict):
                for attr in url_attributes:
                    if attr in result and result[attr]:
                        return result[attr]

            # 尝试从字符串中提取URL
            if isinstance(result, str):
                # 简单的URL提取逻辑
                if result.startswith("http"):
                    return result

            logger.debug(f"无法从结果中提取图片URL: {result}")
            return None

        except Exception as e:
            logger.debug(f"提取图片URL失败: {e}")
            return None

    def _extract_title(self, result) -> Optional[str]:
        """从搜索结果中提取标题"""
        try:
            # 尝试多种可能的属性名
            title_attributes = ["title", "name", "caption", "description", "text"]

            for attr in title_attributes:
                if hasattr(result, attr):
                    title = getattr(result, attr)
                    if title and isinstance(title, str):
                        return title.strip()

            # 如果是字典类型
            if isinstance(result, dict):
                for attr in title_attributes:
                    if attr in result and result[attr]:
                        return str(result[attr]).strip()

            return None

        except Exception as e:
            logger.debug(f"提取标题失败: {e}")
            return None

    def _extract_source(self, result) -> str:
        """从搜索结果中提取来源信息"""
        try:
            # 尝试多种可能的属性名
            source_attributes = ["source", "origin", "platform", "site", "domain"]

            for attr in source_attributes:
                if hasattr(result, attr):
                    source = getattr(result, attr)
                    if source and isinstance(source, str):
                        return source.strip()

            # 如果是字典类型
            if isinstance(result, dict):
                for attr in source_attributes:
                    if attr in result and result[attr]:
                        return str(result[attr]).strip()

            # 尝试从URL中提取域名
            img_url = self._extract_image_url(result)
            if img_url:
                try:
                    domain = urlparse(img_url).netloc
                    if domain:
                        return domain
                except Exception:
                    pass

            return "Ascii2D"

        except Exception as e:
            logger.debug(f"提取来源失败: {e}")
            return "Ascii2D"

    async def get_image_info(self, image_url: str) -> Optional[Dict[str, Any]]:
        """获取图片详细信息"""
        try:
            # 验证URL
            if not self.validate_image_url(image_url):
                return None

            # 这里可以扩展获取更多图片信息的逻辑
            # 比如获取图片尺寸、格式、颜色信息等

            return {
                "url": image_url,
                "domain": urlparse(image_url).netloc,
                "searchable": True,
                "engine": "Ascii2D",
                "max_results": self.max_results,
            }

        except Exception as e:
            logger.error(f"获取图片信息失败: {e}")
            return None

    def validate_image_url(self, url: str) -> bool:
        """验证图片URL是否有效"""
        try:
            parsed = urlparse(url)
            return all([parsed.scheme, parsed.netloc])
        except Exception:
            return False

    async def test_connection(self) -> bool:
        """测试连接是否正常"""
        try:
            # 使用一个简单的测试图片URL来测试服务
            test_url = "https://via.placeholder.com/100x100"
            results = await self.search_by_image_url(test_url)

            # 即使没有结果，只要没有异常就认为连接正常
            logger.info("Ascii2D连接测试成功")
            return True

        except Exception as e:
            logger.error(f"Ascii2D连接测试失败: {e}")
            return False

    def get_service_info(self) -> Dict[str, Any]:
        """获取服务信息"""
        return {
            "name": "Ascii2D",
            "engine": "PicImageSearch",
            "base_url": self.base_url,
            "timeout": 30,  # 硬编码默认值
            "max_results": self.max_results,
            "bovw": self.bovw,
            "version": "3.12.9",
        }

    async def search_multiple_engines(
        self, image_data: bytes, engines: List[str] = None
    ) -> Dict[str, List[ImageSearchResult]]:
        """使用多个搜索引擎进行搜索"""
        if engines is None:
            engines = ["ascii2d"]

        results = {}

        try:
            for engine in engines:
                if engine.lower() == "ascii2d":
                    engine_results = await self.search_by_image_file(image_data)
                    results["ascii2d"] = engine_results
                    logger.info(
                        f"Ascii2D引擎搜索完成，找到 {len(engine_results)} 个结果"
                    )
                # 这里可以扩展其他搜索引擎
                # elif engine.lower() == "saucenao":
                #     # 使用SauceNAO引擎
                #     pass

        except Exception as e:
            logger.error(f"多引擎搜索失败: {e}")

        return results
