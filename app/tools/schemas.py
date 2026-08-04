import re

from pydantic import BaseModel, Field, field_validator


class WebSearchInput(BaseModel):
    query: str = Field(
        description="搜索关键词，简洁明确的中文或英文关键词，如'今天天气'、'Python教程'、'2024年GDP数据'等",
        min_length=2,
        max_length=100,
    )

    @field_validator("query")
    @classmethod
    def clean_query(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 2:
            raise ValueError("搜索关键词太短，请提供至少2个字符")
        if len(v) > 100:
            raise ValueError("搜索关键词过长，请精简到100字以内")
        return v


class CalculatorInput(BaseModel):
    expression: str = Field(
        description="要计算的数学表达式，只包含数字和 + - * / ( ) % 运算符，如 '2 + 3 * 4'、'(10 - 5) / 2'",
        min_length=1,
        max_length=200,
    )

    @field_validator("expression")
    @classmethod
    def validate_expression(cls, v: str) -> str:
        v = v.strip()

        if not v:
            raise ValueError("表达式不能为空，请提供有效的数学表达式")

        # 安全校验：只允许数字和基本运算符
        allowed_pattern = r"^[\d+\-*/().%\s]+$"
        if not re.match(allowed_pattern, v):
            raise ValueError("表达式包含非法字符，只允许数字和 + - * / ( ) % 运算符")

        # 检查括号匹配
        if v.count("(") != v.count(")"):
            raise ValueError("括号不匹配，请检查表达式")

        return v


class KnowledgeSearchInput(BaseModel):
    query: str = Field(
        description="搜索关键词，必须是简洁明确的中文关键词，如'打卡规则'、'迟到规定'、'年假天数'等",
        min_length=2,  # ✅ 改成 2
        max_length=50,
    )

    @field_validator("query")
    @classmethod
    def validate_and_clean(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 2:  # ✅ 改成 < 2
            raise ValueError("搜索关键词太短，请提供至少2个字符的关键词")
        if len(v) > 50:
            raise ValueError("搜索关键词过长，请精简到50字以内")
        return v
