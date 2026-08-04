def calculator(expression: str) -> str:
    """执行数学计算"""
    try:
        # 参数校验
        if not expression or not expression.strip():
            return "错误：表达式不能为空"
        # 安全限制：只允许数字和基本运算符
        allowed = set("0123456789+-*/().% ")
        if not all(c in allowed for c in expression):
            return "错误：表达式包含非法字符"
        result = eval(expression)
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算错误：{str(e)}"
