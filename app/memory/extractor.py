import json
import os
from typing import Any

from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.utils.logger import logger


class UserInfoExtractor:
    """用 LLM 从对话中提取用户信息"""

    def __init__(self):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("请设置环境变量 DEEPSEEK_API_KEY")
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            temperature=0.1,
        )

    def extract(self, messages: list) -> dict[str, Any]:
        """
        从对话中提取用户信息
        返回：{"name": "...", "preferences": [...], "has_new_info": bool}
        """

        # 只取最近1条消息
        recent_messages = messages[-1:]
        logger.log(
            "memory",
            {"operation": "extract", "desc": f"extractor.extract 被调用, 消息数: {len(recent_messages)}"},
        )
        system_prompt = """你是一个信息提取助手。从对话中提取用户的信息。

提取规则：
1. name：用户的名字（如果有）
2. preferences：用户的偏好、喜好（如果有，用列表形式）
3. has_new_info：只要提取到 name 或 preferences 中有新内容，就设为 true，否则设为 false

⚠️ 重要：has_new_info 的值必须是 true 或 false（小写，不加引号），不要用 True/False/"True"/"False"。

返回格式必须是纯 JSON：
{"name": "小明", "preferences": ["辣", "川菜"], "has_new_info": true}"""

        user_content = f"对话内容：{recent_messages}"
        response = self.llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]
        )

        content = str(response.content) if response.content else ""
        logger.log(
            "memory",
            {"operation": "extract", "desc": f"LLM 原始返回长度: {len(content)}"},
        )
        logger.log(
            "memory",
            {"operation": "extract", "desc": f"LLM 原始返回 (repr): {content!r}"},
        )

        try:
            content = content.strip()
            logger.log(
                "memory",
                {"operation": "extract", "desc": f"strip 后: {content!r}"},
            )
            return json.loads(content)
        except (json.JSONDecodeError, ValueError) as e:
            logger.log(
                "error",
                {"operation": "extract", "error": str(e), "desc": "用户信息提取失败"},
            )
            return {"name": "", "preferences": [], "has_new_info": False}
