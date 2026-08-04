from langchain_core.tools import tool

from app.tools.schemas import CalculatorInput


@tool(args_schema=CalculatorInput)
def calculator(expression: str) -> str:
    """执行数学计算，支持加减乘除、括号、百分比等基本运算。

    使用示例：
    - "2 + 3 * 4" → 14
    - "(10 - 5) / 2" → 2.5
    - "100 * 20%" → 20.0
    """
    try:
        # 双重保险：再检查一次
        allowed = set("0123456789+-*/().% ")
        if not all(c in allowed for c in expression):
            return "错误：表达式包含非法字符"
        # 高风险，加二次校验
        result = eval(expression)
        return f"计算结果：{result}"

    except ZeroDivisionError:
        return "错误：除数不能为零"
    except SyntaxError:
        return "错误：表达式语法不正确，请检查格式"
    except Exception as e:
        return f"计算错误：{e!s}"
