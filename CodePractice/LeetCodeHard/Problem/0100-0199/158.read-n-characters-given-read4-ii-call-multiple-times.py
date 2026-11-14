"""
LC 158. Read N Characters Given read4 II - Call Multiple Times — Hard
https://leetcode.com/problems/read-n-characters-given-read4-ii-call-multiple-times/
Best: 缓冲残留字符，多次调用复用 — O(n)
Premium: 用 mock read4 实现
"""

from typing import List


def _make_read4(file_content: str):
    """工厂：返回绑定到 file_content 的 read4(buf4) -> int。"""
    pos = [0]

    def read4(buf4: List[str]) -> int:
        n = 0
        while n < 4 and pos[0] < len(file_content):
            buf4[n] = file_content[pos[0]]
            pos[0] += 1
            n += 1
        return n

    return read4


class Solution:
    """多次 read：内部缓存上次 read4 未消费完的字符。"""

    def __init__(self, read4=None):
        self._read4 = read4
        self._buf4 = [""] * 4
        self._i4 = 0
        self._n4 = 0

    def read(self, buf: List[str], n: int) -> int:
        i = 0
        while i < n:
            if self._i4 == self._n4:
                self._n4 = self._read4(self._buf4)
                self._i4 = 0
                if self._n4 == 0:
                    break
            buf[i] = self._buf4[self._i4]
            self._i4 += 1
            i += 1
        return i


def _test():
    # Example 1: file = "abc", queries read(1), read(2), read(1)
    s = Solution(_make_read4("abc"))
    buf = [""] * 4
    assert s.read(buf, 1) == 1 and "".join(buf[:1]) == "a"
    assert s.read(buf, 2) == 2 and "".join(buf[:2]) == "bc"
    assert s.read(buf, 1) == 0

    # Example 2: file = "abc", read(4) then read(1)
    s = Solution(_make_read4("abc"))
    buf = [""] * 4
    assert s.read(buf, 4) == 3 and "".join(buf[:3]) == "abc"
    assert s.read(buf, 1) == 0

    # Multiple partial reads across read4 boundary
    s = Solution(_make_read4("abcdefghij"))
    buf = [""] * 10
    assert s.read(buf, 3) == 3 and "".join(buf[:3]) == "abc"
    assert s.read(buf, 5) == 5 and "".join(buf[:5]) == "defgh"
    assert s.read(buf, 10) == 2 and "".join(buf[:2]) == "ij"
    print("LC 158 OK")


if __name__ == "__main__":
    _test()
