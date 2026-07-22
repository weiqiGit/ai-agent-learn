import json
import os
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Dict, Any


class UserInfoExtractor:
    """用 LLM 从对话中提取用户信息"""

    def __init__(self):
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1",
            temperature=0.1,
        )

    def extract(self, messages: list) -> Dict[str, Any]:
        """
        从对话中提取用户信息
        返回：{"name": "...", "preferences": [...], "has_new_info": bool}
        """
        # 只取最近 10 条消息
        recent_messages = messages[-1:]

        system_prompt = """你是一个信息提取助手。从对话中提取用户的信息。

提取规则：
1. name：用户的名字（如果有）
2. preferences：用户的偏好、喜好（如果有，用列表形式）
3. has_new_info：判断本次对话是否包含**新的**用户信息（True/False）

注意：
- 只提取**明确**提到的信息，不要猜测
- preferences 只提取具体的偏好，如"辣"、"川菜"、"编程"等
- has_new_info 为 False 时，说明对话中没有新的用户信息

返回格式必须是纯 JSON：
{"name": "小明", "preferences": ["辣", "川菜"], "has_new_info": True}"""

        # ✅ 用消息列表，不是 ChatPromptTemplate
        user_content = f"对话内容：{recent_messages}"
        response = self.llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]
        )

        content = str(response.content)
        print(f"🔍 LLM 原始返回: {content}")

        # ✅ 尝试直接解析
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"❌ 直接解析失败: {e}")
            # 尝试去掉可能的 markdown 代码块
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    pass
            # 如果还是失败，返回默认值
            return {"name": "", "preferences": [], "has_new_info": False}
