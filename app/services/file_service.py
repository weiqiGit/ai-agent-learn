import os
import json
from typing import List, Dict


# 缓存文件路径
CACHE_FILE = "./data/file_cache.json"

# 内存缓存
_file_cache: Dict[str, Dict] = {}


# 加载缓存-在启动时调用
def _load_cache():
    global _file_cache
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                _file_cache = json.load(f)
        except Exception as e:
            print(f"加载文件缓存失败: {e}")
            _file_cache = {}


# 保存缓存到json文件
def _save_cache():
    try:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_file_cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存文件缓存失败: {e}")


# 获取所有已上传文件列表
def get_files_list() -> List[Dict]:
    print(f"缓存中的文件列表: {_file_cache}")
    return list(_file_cache.values())


# 上传文件后更新缓存
def add_file_to_cache(filename: str, source: str, chunks: int):
    _file_cache[filename] = {
        "name": filename,
        "source": source,
        "chunks": chunks,
    }
    _save_cache()
    print(f"缓存已更新: {filename} (chunks={chunks})")


# 删除文件后更新缓存
def remove_file_from_cache(filename: str):
    if filename in _file_cache:
        del _file_cache[filename]
        _save_cache()
        print(f"缓存已移除: {filename}")
    else:
        print(f"缓存中不存在: {filename}")


_load_cache()
print(f"文件列表缓存已加载，共 {len(_file_cache)} 个文件")
