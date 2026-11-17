"""
LC 2630. Memoize II — Hard
https://leetcode.com/problems/memoize-ii/
Best: Identity-keyed nested memo map — O(1) amortized lookup
"""

from typing import Any, Callable, List

def memoize(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Memoize by argument identity (like JS Memoize II)."""
    idx_map: dict = {}
    cache: dict = {}

    def get_idx(obj: Any) -> int:
        key = id(obj)
        if key not in idx_map:
            idx_map[key] = len(idx_map)
        return idx_map[key]

    def wrapped(*params: Any) -> Any:
        key = ",".join(str(get_idx(p)) for p in params)
        if key not in cache:
            cache[key] = fn(*params)
        return cache[key]

    wrapped.cache = cache  # type: ignore
    return wrapped

class Solution:
    def memoize(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        return memoize(fn)

def _test():

    calls = {"n": 0}

    def add(a, b):
        calls["n"] += 1
        return a + b

    f = memoize(add)
    assert f(2, 3) == 5
    assert f(2, 3) == 5
    assert calls["n"] == 1
    o = []
    def same(x):
        calls["n"] += 1
        return x
    g = memoize(same)
    assert g(o) is o and g(o) is o
    assert calls["n"] == 2  # one for add's first, one for same
    print("LC 2630 OK")

if __name__ == "__main__":
    _test()
