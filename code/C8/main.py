"""
RAG系统主程序
"""

import os
import sys
import io
import logging
from pathlib import Path
from typing import List

BASE_DIR = Path(__file__).resolve().parent

# 强制标准输入输出使用 UTF-8，避免中文 / emoji / LLM 输出触发 ascii 编码错误
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stdin, "buffer"):
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace")

# 添加模块路径
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
from config import DEFAULT_CONFIG, RAGConfig
from rag_modules import (
    DataPreparationModule,
    IndexConstructionModule,
    RetrievalOptimizationModule,
    GenerationIntegrationModule
)

# 加载环境变量
load_dotenv(BASE_DIR / ".env")


def safe_print(text: str = "", end: str = "\n", flush: bool = False):
    """在终端编码不友好时也尽量安全输出文本。"""
    output = f"{text}{end}"

    try:
        sys.stdout.write(output)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        safe_output = output.encode(encoding, errors="replace").decode(encoding, errors="replace")
        sys.stdout.write(safe_output)

    if flush:
        sys.stdout.flush()


class SafeTextStream:
    """对终端编码不友好的输出流做兜底封装。"""

    def __init__(self, stream):
        self.stream = stream

    def write(self, text):
        try:
            return self.stream.write(text)
        except UnicodeEncodeError:
            encoding = getattr(self.stream, "encoding", None) or "utf-8"
            safe_text = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
            return self.stream.write(safe_text)

    def flush(self):
        return self.stream.flush()


def safe_input(prompt: str = "") -> str:
    """在终端编码不友好时安全展示提示并读取输入。"""
    if prompt:
        safe_print(prompt, end="", flush=True)

    if hasattr(sys.stdin, "buffer"):
        raw_input = sys.stdin.buffer.readline()
        if raw_input == b"":
            raise EOFError
        try:
            user_input = raw_input.decode("utf-8")
        except UnicodeDecodeError:
            encoding = sys.stdin.encoding or "utf-8"
            user_input = raw_input.decode(encoding, errors="replace")
    else:
        user_input = sys.stdin.readline()
        if user_input == "":
            raise EOFError

    if user_input == "":
        raise EOFError

    return user_input.rstrip("\n")


# 配置日志
handler = logging.StreamHandler(SafeTextStream(sys.stderr))
handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
))

logging.basicConfig(
    level=logging.INFO,
    handlers=[handler],
    force=True
)

logger = logging.getLogger(__name__)


class RecipeRAGSystem:
    """食谱RAG系统主类"""

    def __init__(self, config: RAGConfig = None):
        """
        初始化RAG系统

        Args:
            config: RAG系统配置，默认使用DEFAULT_CONFIG
        """
        self.config = config or DEFAULT_CONFIG
        self.data_module = None
        self.index_module = None
        self.retrieval_module = None
        self.generation_module = None

        # 检查数据路径
        if not Path(self.config.data_path).exists():
            raise FileNotFoundError(f"数据路径不存在: {self.config.data_path}")

        # 检查API密钥
        if not os.getenv("DEEPSEEK_API_KEY"):
            raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量")

    @staticmethod
    def _emit(message: str, verbose: bool = True, status_callback=None):
        """统一处理状态输出。"""
        if status_callback is not None:
            status_callback(message)
        if verbose:
            safe_print(message)

    def initialize_system(self, verbose: bool = True, status_callback=None):
        """初始化所有模块"""
        self._emit("🚀 正在初始化RAG系统...", verbose=verbose, status_callback=status_callback)

        # 1. 初始化数据准备模块
        self._emit("初始化数据准备模块...", verbose=verbose, status_callback=status_callback)
        self.data_module = DataPreparationModule(self.config.data_path)

        # 2. 初始化索引构建模块
        self._emit("初始化索引构建模块...", verbose=verbose, status_callback=status_callback)
        self.index_module = IndexConstructionModule(
            model_name=self.config.embedding_model,
            index_save_path=self.config.index_save_path
        )

        # 3. 初始化生成集成模块
        self._emit("🤖 初始化生成集成模块...", verbose=verbose, status_callback=status_callback)
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        self._emit("✅ 系统初始化完成！", verbose=verbose, status_callback=status_callback)

    def build_knowledge_base(self, verbose: bool = True, status_callback=None):
        """构建知识库"""
        self._emit("\n正在构建知识库...", verbose=verbose, status_callback=status_callback)

        # 1. 尝试加载已保存的索引
        vectorstore = self.index_module.load_index()

        if vectorstore is not None:
            self._emit("✅ 成功加载已保存的向量索引！", verbose=verbose, status_callback=status_callback)

            # 仍需要加载文档和分块用于检索模块
            self._emit("加载食谱文档...", verbose=verbose, status_callback=status_callback)
            self.data_module.load_documents()

            self._emit("进行文本分块...", verbose=verbose, status_callback=status_callback)
            chunks = self.data_module.chunk_documents()
        else:
            self._emit("未找到已保存的索引，开始构建新索引...", verbose=verbose, status_callback=status_callback)

            # 2. 加载文档
            self._emit("加载食谱文档...", verbose=verbose, status_callback=status_callback)
            self.data_module.load_documents()

            # 3. 文本分块
            self._emit("进行文本分块...", verbose=verbose, status_callback=status_callback)
            chunks = self.data_module.chunk_documents()

            # 4. 构建向量索引
            self._emit("构建向量索引...", verbose=verbose, status_callback=status_callback)
            vectorstore = self.index_module.build_vector_index(chunks)

            # 5. 保存索引
            self._emit("保存向量索引...", verbose=verbose, status_callback=status_callback)
            self.index_module.save_index()

        # 6. 初始化检索优化模块
        self._emit("初始化检索优化...", verbose=verbose, status_callback=status_callback)
        self.retrieval_module = RetrievalOptimizationModule(vectorstore, chunks)

        # 7. 显示统计信息
        stats = self.data_module.get_statistics()
        self._emit("\n📊 知识库统计:", verbose=verbose, status_callback=status_callback)
        self._emit(f"   文档总数: {stats['total_documents']}", verbose=verbose, status_callback=status_callback)
        self._emit(f"   文本块数: {stats['total_chunks']}", verbose=verbose, status_callback=status_callback)
        self._emit(f"   菜品分类: {list(stats['categories'].keys())}", verbose=verbose, status_callback=status_callback)
        self._emit(f"   难度分布: {stats['difficulties']}", verbose=verbose, status_callback=status_callback)

        self._emit("✅ 知识库构建完成！", verbose=verbose, status_callback=status_callback)

    def ask_question(self, question: str, stream: bool = False, verbose: bool = True, status_callback=None):
        """
        回答用户问题

        Args:
            question: 用户问题
            stream: 是否使用流式输出

        Returns:
            生成的回答或生成器
        """
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")

        self._emit(f"\n❓ 用户问题: {question}", verbose=verbose, status_callback=status_callback)

        normalized_question = self.generation_module.normalize_query_text(question)
        if normalized_question != question:
            self._emit(f"🧹 查询已规范化: {normalized_question}", verbose=verbose, status_callback=status_callback)

        # 1. 查询路由
        route_type = self.generation_module.query_router(normalized_question)
        self._emit(f"🎯 查询类型: {route_type}", verbose=verbose, status_callback=status_callback)

        # 2. 智能查询重写
        if route_type in {"list", "detail"}:
            rewritten_query = normalized_question

            if route_type == "list":
                self._emit(f"📝 列表查询保持原样: {normalized_question}", verbose=verbose, status_callback=status_callback)
            else:
                self._emit(f"📝 详情查询保持原样: {normalized_question}", verbose=verbose, status_callback=status_callback)
        else:
            self._emit("🤖 智能分析查询...", verbose=verbose, status_callback=status_callback)
            rewritten_query = self.generation_module.query_rewrite(normalized_question)

        # 3. 检索相关子块
        self._emit("🔍 检索相关文档...", verbose=verbose, status_callback=status_callback)
        filters = self._extract_filters_from_query(normalized_question)

        if filters:
            self._emit(f"应用过滤条件: {filters}", verbose=verbose, status_callback=status_callback)
            relevant_chunks = self.retrieval_module.metadata_filtered_search(
                rewritten_query,
                filters,
                top_k=self.config.top_k
            )
        else:
            relevant_chunks = self.retrieval_module.hybrid_search(
                rewritten_query,
                top_k=self.config.top_k
            )

        # 显示检索到的子块信息
        if relevant_chunks:
            chunk_info = []

            for chunk in relevant_chunks:
                dish_name = chunk.metadata.get("dish_name", "未知菜品")
                content_preview = chunk.page_content[:100].strip()

                if content_preview.startswith("#"):
                    title_end = content_preview.find("\n") if "\n" in content_preview else len(content_preview)
                    section_title = content_preview[:title_end].replace("#", "").strip()
                    chunk_info.append(f"{dish_name}({section_title})")
                else:
                    chunk_info.append(f"{dish_name}(内容片段)")

            self._emit(
                f"找到 {len(relevant_chunks)} 个相关文档块: {', '.join(chunk_info)}",
                verbose=verbose,
                status_callback=status_callback,
            )
        else:
            self._emit(f"找到 {len(relevant_chunks)} 个相关文档块", verbose=verbose, status_callback=status_callback)

        # 4. 检查是否找到相关内容
        if not relevant_chunks:
            return "抱歉，没有找到相关的食谱信息。请尝试其他菜品名称或关键词。"

        # 5. 根据路由类型选择回答方式
        if route_type == "list":
            self._emit("📋 生成菜品列表...", verbose=verbose, status_callback=status_callback)
            relevant_docs = self.data_module.get_parent_documents(relevant_chunks)

            doc_names = []
            for doc in relevant_docs:
                dish_name = doc.metadata.get("dish_name", "未知菜品")
                doc_names.append(dish_name)

            if doc_names:
                self._emit(f"找到文档: {', '.join(doc_names)}", verbose=verbose, status_callback=status_callback)

            return self.generation_module.generate_list_answer(question, relevant_docs)

        # 详细查询：获取完整文档并生成详细回答
        self._emit("获取完整文档...", verbose=verbose, status_callback=status_callback)
        relevant_docs = self.data_module.get_parent_documents(relevant_chunks)

        doc_names = []
        for doc in relevant_docs:
            dish_name = doc.metadata.get("dish_name", "未知菜品")
            doc_names.append(dish_name)

        if doc_names:
            self._emit(f"找到文档: {', '.join(doc_names)}", verbose=verbose, status_callback=status_callback)
        else:
            self._emit(f"对应 {len(relevant_docs)} 个完整文档", verbose=verbose, status_callback=status_callback)

        self._emit("✍️ 生成详细回答...", verbose=verbose, status_callback=status_callback)

        if route_type == "detail":
            if stream:
                return self.generation_module.generate_step_by_step_answer_stream(
                    normalized_question,
                    relevant_docs
                )

            return self.generation_module.generate_step_by_step_answer(
                normalized_question,
                relevant_docs
            )

        if stream:
            return self.generation_module.generate_basic_answer_stream(
                normalized_question,
                relevant_docs
            )

        return self.generation_module.generate_basic_answer(
            normalized_question,
            relevant_docs
        )

    def _extract_filters_from_query(self, query: str) -> dict:
        """从用户问题中提取元数据过滤条件。"""
        filters = {}

        # 分类关键词
        category_keywords = DataPreparationModule.get_supported_categories()
        for cat in category_keywords:
            if cat in query:
                filters["category"] = cat
                break

        # 难度关键词
        difficulty_keywords = DataPreparationModule.get_supported_difficulties()
        for diff in sorted(difficulty_keywords, key=len, reverse=True):
            if diff in query:
                filters["difficulty"] = diff
                break

        return filters

    def search_by_category(self, category: str, query: str = "") -> List[str]:
        """
        按分类搜索菜品

        Args:
            category: 菜品分类
            query: 可选的额外查询条件

        Returns:
            菜品名称列表
        """
        if not self.retrieval_module:
            raise ValueError("请先构建知识库")

        search_query = query if query else category
        filters = {"category": category}

        docs = self.retrieval_module.metadata_filtered_search(
            search_query,
            filters,
            top_k=10
        )

        dish_names = []
        for doc in docs:
            dish_name = doc.metadata.get("dish_name", "未知菜品")
            if dish_name not in dish_names:
                dish_names.append(dish_name)

        return dish_names

    def get_ingredients_list(self, dish_name: str) -> str:
        """
        获取指定菜品的食材信息

        Args:
            dish_name: 菜品名称

        Returns:
            食材信息
        """
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")

        docs = self.retrieval_module.hybrid_search(dish_name, top_k=3)

        answer = self.generation_module.generate_basic_answer(
            f"{dish_name}需要什么食材？",
            docs
        )

        return answer

    def run_interactive(self):
        """运行交互式问答"""
        safe_print("=" * 60)
        safe_print("🍽️  尝尝咸淡RAG系统 - 交互式问答  🍽️")
        safe_print("=" * 60)
        safe_print("💡 解决您的选择困难症，告别'今天吃什么'的世纪难题！")

        # 初始化系统
        self.initialize_system()

        # 构建知识库
        self.build_knowledge_base()

        safe_print("\n交互式问答 (输入'退出'结束):")

        while True:
            try:
                user_input = safe_input("\n您的问题: ").strip()

                if user_input.lower() in ["退出", "quit", "exit", ""]:
                    break

                stream_choice = safe_input("是否使用流式输出? (y/n, 默认y): ").strip().lower()
                use_stream = stream_choice != "n"

                safe_print("\n回答:")

                if use_stream:
                    answer_or_stream = self.ask_question(user_input, stream=True)

                    if isinstance(answer_or_stream, str):
                        safe_print(f"{answer_or_stream}\n")
                    else:
                        for chunk in answer_or_stream:
                            safe_print(chunk, end="", flush=True)
                        safe_print("\n")
                else:
                    answer = self.ask_question(user_input, stream=False)
                    safe_print(f"{answer}\n")

            except KeyboardInterrupt:
                break
            except EOFError:
                break
            except Exception as e:
                safe_print(f"处理问题时出错: {str(e)}")

        safe_print("\n感谢使用尝尝咸淡RAG系统！")


def main():
    """主函数"""
    try:
        rag_system = RecipeRAGSystem()
        rag_system.run_interactive()

    except Exception as e:
        logger.error(f"系统运行出错: {str(e)}")
        safe_print(f"系统错误: {str(e)}")


if __name__ == "__main__":
    main()
