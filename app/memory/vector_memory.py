import json
import os
from datetime import datetime
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


class VectorMemory:
    """自由文本 + 向量检索 的长期记忆（模糊信息）"""

    # ✅ 用智谱替代 HuggingFace
    def __init__(self, persist_dir: str = "./user_memory_db"):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        api_key = os.getenv("ZHIPUAI_API_KEY")

        if not api_key:
            raise ValueError("请设置环境变量 DEEPSEEK_API_KEY")

        self.embeddings = ZhipuAIEmbeddings(model="embedding-2", api_key=api_key)

        self.llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=SecretStr(api_key),
            base_url="https://api.deepseek.com/v1",
            temperature=0.1,
        )

    def _get_collection_name(self, user_id: str) -> str:
        return f"user_memory_{user_id}"

    def _get_vector_store(self, user_id: str, texts: Optional[List[str]] = None):
        """获取或创建用户向量库"""
        collection_name = self._get_collection_name(user_id)

        if texts:
            return Chroma.from_texts(
                texts=texts,
                embedding=self.embeddings,
                collection_name=collection_name,
                persist_directory=self.persist_dir,
            )
        else:
            return Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_dir,
            )

    def _summarize(self, conversation_text: str) -> str:
        """用 LLM 生成摘要（模糊信息）"""
        prompt = f"""请从以下对话中提取用户的个人信息，用一段话概括（不超过50字）。
只提取用户明确提到的信息，不要猜测。

对话：{conversation_text}

只输出一段话，不要有其他内容。"""

        try:
            response = self.llm.invoke(
                [
                    SystemMessage(content="你是一个信息提取助手。"),
                    HumanMessage(content=prompt),
                ]
            )
            summary = str(response.content).strip()
            print(f"🔍 生成的摘要: {summary}")
            return summary if len(summary) > 5 else ""
        except Exception as e:
            print(f"❌ 生成摘要失败: {e}")
            return ""

    def store(self, user_id: str, conversation_text: str) -> None:
        """存储用户记忆（自由文本）"""
        summary = self._summarize(conversation_text)
        if not summary:
            return

        timestamp = datetime.now().isoformat()
        full_text = json.dumps(
            {"user_id": user_id, "timestamp": timestamp, "content": summary},
            ensure_ascii=False,
        )

        try:
            vectordb = self._get_vector_store(user_id)
            existing = vectordb.similarity_search(summary, k=1)
            if existing and existing[0].page_content == summary:
                print(f"⏭️ 记忆已存在: {summary[:30]}...")
                return
        except Exception:
            pass

        vectordb = self._get_vector_store(user_id, [full_text])
        print(f"✅ 向量记忆已存储: {summary[:50]}...")

    def retrieve(self, user_id: str, query: str, k: int = 3) -> List[str]:
        """检索用户记忆"""
        try:
            vectordb = self._get_vector_store(user_id)
            # 先检索 k*2 条，确保有足够的候选
            results = vectordb.similarity_search(query, k=k * 2)
            # ✅ 解析时间戳，按时间降序排序
            parsed_results = []
            for doc in results:
                try:
                    data = json.loads(doc.page_content)
                    # 验证 user_id 匹配
                    if data.get("user_id") == user_id:
                        parsed_results.append(
                            {
                                "content": data.get("content", ""),
                                "timestamp": data.get("timestamp", ""),
                                "text": doc.page_content,
                            }
                        )
                except json.JSONDecodeError:
                    # 兼容旧格式（没有 JSON 包装）
                    parsed_results.append(
                        {
                            "content": doc.page_content,
                            "timestamp": "",
                            "text": doc.page_content,
                        }
                    )

            # ✅ 按时间戳降序排序（最新的在前）
            parsed_results.sort(key=lambda x: x["timestamp"], reverse=True)

            # 返回前 k 条的内容
            return [r["content"] for r in parsed_results[:k]]
        except Exception as e:
            print(f"⚠️ 检索失败: {e}")
            return []

    def get_context_prompt(self, user_id: str, query: str) -> str:
        """生成用于注入 Prompt 的记忆文本"""
        memories = self.retrieve(user_id, query, k=3)
        if not memories:
            return ""
        return "用户相关记忆：\n" + "\n".join([f"- {m}" for m in memories])
