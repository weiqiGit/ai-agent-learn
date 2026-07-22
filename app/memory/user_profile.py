import json
import os
from typing import Dict, Any
from datetime import datetime


def has_potential_info(content: str) -> bool:
    """快速判断单条消息是否可能包含用户信息"""
    keywords = [
        "我叫",
        "我是",
        "我是一名",
        "我喜欢",
        "我讨厌",
        "我的",
        "我今年",
        "我不喜欢",
    ]
    print(f"写时提取已触发:: {content}")
    return any(kw in content for kw in keywords)


class UserProfileMemory:
    """用户画像存储（生产级：增量更新）"""

    def __init__(self, storage_dir: str = "./user_profiles"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

    def _get_file_path(self, user_id: str) -> str:
        return os.path.join(self.storage_dir, f"{user_id}.json")

    def get(self, user_id: str) -> Dict[str, Any]:
        file_path = self._get_file_path(user_id)
        if not os.path.exists(file_path):
            return {}
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def merge(self, user_id: str, new_info: Dict[str, Any]) -> bool:
        """
        增量合并新信息到用户画像
        返回：是否有实际更新
        """
        profile = self.get(user_id)
        updated = False

        # 合并 name
        if "name" in new_info and new_info["name"]:
            if profile.get("name") != new_info["name"]:
                profile["name"] = new_info["name"]
                updated = True

        # 合并 preferences
        if "preferences" in new_info and new_info["preferences"]:
            if "preferences" not in profile:
                profile["preferences"] = []
            for pref in new_info["preferences"]:
                if pref not in profile["preferences"]:
                    profile["preferences"].append(pref)
                    updated = True

        if updated:
            profile["last_updated"] = datetime.now().isoformat()
            with open(self._get_file_path(user_id), "w", encoding="utf-8") as f:
                json.dump(profile, f, ensure_ascii=False, indent=2)

        return updated

    def get_context_prompt(self, user_id: str) -> str:
        """生成用于注入 Prompt 的用户画像文本"""
        profile = self.get(user_id)
        if not profile:
            return ""

        parts = []
        if "name" in profile:
            parts.append(f"用户姓名：{profile['name']}")
        if "preferences" in profile and profile["preferences"]:
            parts.append(f"用户偏好：{', '.join(profile['preferences'])}")

        if parts:
            return "\n".join(parts)
        return ""
