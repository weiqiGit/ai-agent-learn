# test_tools.py - 放在项目根目录

import os
import sys

# 确保项目根目录在 Python 路径里
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 加载环境变量（如果你用 .env 文件的话）
from dotenv import load_dotenv

load_dotenv()

# 然后再 import 你的 tool
from app.tools import calculator, knowledge_search, web_search


def test_knowledge_search():
    print("=" * 50)
    print("测试 knowledge_search")
    print("=" * 50)

    # 测试 1：正常调用
    print("\n1. 测试正常调用:")
    try:
        result = knowledge_search.invoke({"query": "打卡规则"})
        print(f"   结果: {result[:200]}")
    except Exception as e:
        print(f"   错误: {e}")

    # 测试 2：空字符串（应该校验失败）
    print("\n2. 测试空字符串:")
    try:
        knowledge_search.invoke({"query": ""})
        print("   ❌ 没报错，校验没生效")
    except Exception as e:
        print(f"   ✅ 校验失败（预期）: {e}")

    # 测试 3：单字符（应该校验失败）
    print("\n3. 测试单字符:")
    try:
        knowledge_search.invoke({"query": "打"})
        print("   ❌ 没报错，校验没生效")
    except Exception as e:
        print(f"   ✅ 校验失败（预期）: {e}")


def test_calculator():
    print("\n" + "=" * 50)
    print("测试 calculator")
    print("=" * 50)

    # 测试 1：正常计算
    print("\n1. 测试正常计算:")
    try:
        result = calculator.invoke({"expression": "2 + 3 * 4"})
        print(f"   结果: {result}")
    except Exception as e:
        print(f"   错误: {e}")

    # 测试 2：非法字符
    print("\n2. 测试非法字符:")
    try:
        calculator.invoke({"expression": "__import__('os')"})
        print("   ❌ 没报错，校验没生效")
    except Exception as e:
        print(f"   ✅ 校验失败（预期）: {e}")


def test_web_search():
    print("\n" + "=" * 50)
    print("测试 web_search")
    print("=" * 50)

    print("\n1. 测试正常搜索:")
    try:
        result = web_search.invoke({"query": "今天北京天气"})
        print(f"   结果: {result[:200]}")
    except Exception as e:
        print(f"   错误: {e}")


if __name__ == "__main__":
    test_knowledge_search()
    test_calculator()
    test_web_search()
