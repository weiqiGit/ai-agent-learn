import random
import sqlite3
from datetime import datetime, timedelta, timezone

from app.tools.schemas import SCHEMA_INFO

DB_PATH = "./data.db"


def init_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. 创建订单表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            city TEXT NOT NULL,
            category TEXT NOT NULL,
            order_amount REAL NOT NULL,
            order_status TEXT NOT NULL,
            create_date TEXT NOT NULL,
            pay_channel TEXT NOT NULL
        )
    """)

    # 2. 插入测试数据
    categories = ["数码", "服饰", "生鲜", "家电", "图书", "美妆", "食品"]
    cities = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉"]
    statuses = ["paid", "refunded", "cancelled"]
    channels = ["微信", "支付宝", "银行卡"]

    orders = []
    start_date = datetime(2026, 1, 1, tzinfo=timezone.utc)

    for i in range(100):
        order_id = f"ORD{str(i + 1).zfill(6)}"
        user_id = f"USER{random.randint(1, 20):04d}"
        city = random.choice(cities)
        category = random.choice(categories)
        order_amount = round(random.uniform(10, 5000), 2)
        order_status = random.choice(statuses)
        days = random.randint(0, 365)
        create_date = (start_date + timedelta(days=days)).strftime("%Y-%m-%d")
        pay_channel = random.choice(channels)
        orders.append(
            (
                order_id,
                user_id,
                city,
                category,
                order_amount,
                order_status,
                create_date,
                pay_channel,
            )
        )

    cursor.executemany(
        """
        INSERT INTO orders (order_id, user_id, city, category, order_amount, order_status, create_date, pay_channel)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        orders,
    )

    conn.commit()
    conn.close()

    print(f"✅ 数据库已创建：{DB_PATH}")
    print("   - orders: 100 条订单")
    print("\n📋 字段说明：")
    print(SCHEMA_INFO)


if __name__ == "__main__":
    init_database()
