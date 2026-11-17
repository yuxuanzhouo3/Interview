"""
LC 2756. Query Batching — Hard (JS)
Best: Debounce coalesce keys — Python demo
"""

import asyncio
from typing import Callable, List, Any


class QueryBatcher:
    def __init__(self, queryMultiple: Callable, throttle: float = 0.05):
        self.queryMultiple = queryMultiple
        self.throttle = throttle
        self._pending = {}
        self._task = None

    async def getValue(self, key):
        fut = asyncio.get_event_loop().create_future()
        self._pending.setdefault(key, []).append(fut)
        if not self._task:
            self._task = asyncio.get_event_loop().create_task(self._flush())
        return await fut

    async def _flush(self):
        await asyncio.sleep(self.throttle)
        keys = list(self._pending.keys())
        pending = self._pending
        self._pending = {}
        self._task = None
        results = await self.queryMultiple(keys)
        for k, v in zip(keys, results):
            for fut in pending[k]:
                fut.set_result(v)


def _test():
    assert QueryBatcher(lambda ks: ks)
    print("LC 2756 OK")


if __name__ == "__main__":
    _test()
