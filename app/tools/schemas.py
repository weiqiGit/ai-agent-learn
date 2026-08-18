import re

from pydantic import BaseModel, Field, field_validator


# ==================== SQL 查询输入模型 ====================
class SQLQueryInput(BaseModel):
    question: str = Field(
        description="用户用自然语言提出的数据查询问题",
        min_length=2,
        max_length=200,
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 2:
            raise ValueError("查询内容太短，请提供更具体的问题")
        if len(v) > 200:
            raise ValueError("查询内容过长，请精简到200字以内")
        return v


# ==================== 数据库 Schema 描述 ====================
SCHEMA_INFO = """
表名：orders（订单表，SQLite 数据库）
字段说明：
  - order_id: 订单ID（唯一标识，格式：ORD000001）
  - user_id: 用户ID（格式：USER0001）
  - city: 下单城市（如：北京、上海、广州、深圳、杭州、成都、武汉）
  - category: 商品品类，取值包括：
    * 数码（手机、电脑、耳机等电子产品）
    * 服饰（衣服、鞋子、包等穿戴类）
    * 生鲜（水果、蔬菜、肉类等食品）
    * 家电（冰箱、电视、洗衣机等家用电器）
    * 图书（书籍、杂志等出版物）
    * 美妆（化妆品、护肤品等美容产品）
    * 食品（零食、饮料等日常食品）
  - order_amount: 订单金额（单位：元，保留两位小数）
  - order_status: 订单状态，取值包括：
    * paid（已支付）
    * refunded（已退款）
    * cancelled（已取消）
  - create_date: 下单日期（格式：YYYY-MM-DD）
  - pay_channel: 支付渠道，取值包括：微信、支付宝、银行卡
"""


class WebSearchInput(BaseModel):
    query: str = Field(
        description="搜索互联网上的最新信息、新闻、实时数据。当需要查询外部信息时使用。搜索关键词，简洁明确的中文或英文关键词，如'今天天气'、'Python教程'、'2024年GDP数据'等",
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
        description="执行数学计算，如加减乘除、百分比等。当用户需要计算时使用。要计算的数学表达式，只包含数字和 + - * / ( ) % 运算符，如 '2 + 3 * 4'、'(10 - 5) / 2'",
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
        description="查询公司内部文档、制度、流程、产品手册等。当用户问关于公司内部的事时使用。搜索关键词，必须是简洁明确的中文关键词，如'打卡规则'、'迟到规定'、'年假天数'等",
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


class ApprovalRequest(BaseModel):
    thread_id: str
