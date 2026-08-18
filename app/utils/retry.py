# app/utils/retry.py
import asyncio
import random
import time
from collections.abc import Callable
from functools import wraps
from typing import Optional


def retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff: float = 2.0,
    max_delay: float = 30.0,
    jitter: float = 0.2,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable] = None,
):
    """
    统一重试装饰器（指数退避 + 随机抖动）

    Args:
        max_retries: 最大重试次数
        base_delay: 初始延迟（秒）
        backoff: 退避倍数（每次翻倍）
        max_delay: 最大延迟上限（秒）
        jitter: 抖动比例（0.2 表示 ±20%）
        exceptions: 需要重试的异常类型
        on_retry: 重试时的回调函数
    """

    def decorator(func):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            current_delay = base_delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt >= max_retries:
                        raise  # ✅ 最后一次直接抛出原始异常

                    if on_retry:
                        on_retry(func.__name__, attempt + 1, e, current_delay)

                    jitter_amount = current_delay * jitter
                    actual_delay = current_delay + random.uniform(
                        -jitter_amount, jitter_amount
                    )
                    actual_delay = max(0.1, min(actual_delay, max_delay))

                    time.sleep(actual_delay)
                    current_delay *= backoff

            # 不会执行到这里
            return None

        # async 版本同样处理
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            current_delay = base_delay

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt >= max_retries:
                        raise  # ✅ 最后一次直接抛出

                    if on_retry:
                        on_retry(func.__name__, attempt + 1, e, current_delay)

                    jitter_amount = current_delay * jitter
                    actual_delay = current_delay + random.uniform(
                        -jitter_amount, jitter_amount
                    )
                    actual_delay = max(0.1, min(actual_delay, max_delay))

                    await asyncio.sleep(actual_delay)
                    current_delay *= backoff

            return None

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
