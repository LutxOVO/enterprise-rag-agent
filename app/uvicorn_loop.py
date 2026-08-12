"""Windows 下为 Uvicorn 提供 psycopg 兼容的事件循环工厂。"""

import asyncio


def selector_loop_factory():
    """创建 SelectorEventLoop，避免 Windows Proactor 与 psycopg 冲突。

    这里作为 Uvicorn 的自定义 ``--loop`` 导入路径传入。Uvicorn 会把这个
    函数直接交给 ``asyncio.run(..., loop_factory=...)``，所以必须返回已经
    创建好的事件循环实例，而不是返回事件循环类。
    """
    return asyncio.SelectorEventLoop()
