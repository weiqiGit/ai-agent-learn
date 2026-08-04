import json
import os
import re
from datetime import datetime, timezone
from typing import Any

from app.utils.logger import logger


def has_potential_info(content: str) -> bool:
    """快速判断单条消息是否可能包含用户信息"""
    pattern = r"我(叫|是|是一名|喜欢|讨厌|爱|打算|计划|最近|经常|在学|在做|想).{2,20}"
    return re.search(pattern, content) is not None


class UserProfileMemory:
    """用户画像存储（生产级：增量更新）"""

    def __init__(self, storage_dir: str = "./user_profiles"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        print("⚠️ 文件不存在啦啦啦")  # 👈 加这行

    def _get_file_path(self, user_id: str) -> str:
        return os.path.join(self.storage_dir, f"{user_id}.json")

    def get(self, user_id: str) -> dict[str, Any]:
        try:
            file_path = self._get_file_path(user_id)

            print(f"📂 尝试读取: {file_path}")  # 👈 加这行看路径
            if not os.path.exists(file_path):
                print("⚠️ 文件不存在，返回空字典")  # 👈 加这行
                return {}
            print("✅ 文件存在，尝试解析 JSON")  # 👈 加这行
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)

        except Exception as e:
            logger.log(
                "error",
                {
                    "user_id": user_id,
                    "operation": "get_profile",
                    "error": str(e),
                    "desc": f"用户 {user_id} 画像读取失败",
                },
            )
            return {}

    def merge(self, user_id: str, new_info: dict[str, Any]) -> bool:
        """
        增量合并新信息到用户画像
        返回：是否有实际更新
        """
        try:
            print(
                f"🔍 merge 被调用, user_id: {user_id}, new_info: {new_info}"
            )  # ← 加这行
            profile = self.get(user_id)
            print(f"profile:{profile}")  # ← 加这行
            updated = False

            # 合并 name
            if (
                "name" in new_info
                and new_info["name"]
                and profile.get("name") != new_info["name"]
            ):
                profile["name"] = new_info["name"]
                # ✅ 清空 preferences
                if "preferences" in profile:
                    profile["preferences"] = []
                updated = True

            # 合并 preferences
            if new_info.get("preferences"):
                if "preferences" not in profile:
                    profile["preferences"] = []
                for pref in new_info["preferences"]:
                    if pref not in profile["preferences"]:
                        profile["preferences"].append(pref)
                        updated = True

            if updated:
                profile["last_updated"] = datetime.now(timezone.utc).isoformat()
                with open(self._get_file_path(user_id), "w", encoding="utf-8") as f:
                    json.dump(profile, f, ensure_ascii=False, indent=2)
                # 记录-标签记忆存储成功
                logger.log(
                    "memory",
                    {
                        "user_id": user_id,
                        "memory_type": "label",
                        "desc": f"用户画像已更新: user_id: {user_id}, new_info: {new_info}",
                    },
                )

            return updated
        except IOError as e:
            logger.log(
                "error",
                {
                    "user_id": user_id,
                    "operation": "merge",
                    "error": f"文件读写失败: {str(e)}",
                    "desc": f"用户 {user_id} 画像写入失败",
                },
            )
            return False
        except json.JSONDecodeError as e:
            logger.log(
                "error",
                {
                    "user_id": user_id,
                    "operation": "merge",
                    "error": f"JSON 解析失败: {str(e)}",
                    "desc": f"用户 {user_id} 画像数据损坏",
                },
            )
            return False
        except Exception as e:
            logger.log(
                "error",
                {
                    "user_id": user_id,
                    "operation": "merge",
                    "error": str(e),
                    "desc": f"用户 {user_id} 画像合并失败",
                },
            )
            return False

    def get_context_prompt(self, user_id: str) -> str:
        """生成用于注入 Prompt 的用户画像文本"""
        profile = self.get(user_id)
        if not profile:
            return ""

        parts = []
        if "name" in profile:
            parts.append(f"用户姓名：{profile['name']}")
        if profile.get("preferences"):
            parts.append(f"用户偏好：{', '.join(profile['preferences'])}")

        if parts:
            return "\n".join(parts)
        return ""
