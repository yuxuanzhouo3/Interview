"""
LC 1585. Check If String Is Transformable With Substring Sort Operations — Hard
https://leetcode.com/problems/check-if-string-is-transformable-with-substring-sort-operations/
Best: Queues of indices — bubble sort intuition — O(n)
"""

from collections import deque


class Solution:
    def isTransformable(self, s: str, t: str) -> bool:
        pos = [deque() for _ in range(10)]
        for i, ch in enumerate(s):
            pos[int(ch)].append(i)
        for ch in t:
            d = int(ch)
            if not pos[d]:
                return False
            idx = pos[d].popleft()
            for smaller in range(d):
                if pos[smaller] and pos[smaller][0] < idx:
                    return False
        return True


def _test():
    s = Solution()
    assert s.isTransformable("84532", "34852") is True
    assert s.isTransformable("34521", "23415") is True
    assert s.isTransformable("12345", "12435") is False
    print("LC 1585 OK")


if __name__ == "__main__":
    _test()
