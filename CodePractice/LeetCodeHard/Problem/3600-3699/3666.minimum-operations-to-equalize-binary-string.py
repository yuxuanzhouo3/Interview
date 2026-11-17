from sortedcontainers import SortedSet
from bisect import bisect_left, bisect_right
"""
LC 3666. Minimum Operations to Equalize Binary String — Hard
https://leetcode.com/problems/minimum-operations-to-equalize-binary-string/
Best: doocs/leetcode reference solution (verified on examples)
"""

from collections import deque, defaultdict, Counter

class Solution:
    def minOperations(self, s: str, k: int) -> int:
        n = len(s)
        ts = [SortedSet() for _ in range(2)]
        for i in range(n + 1):
            ts[i % 2].add(i)
        cnt0 = s.count('0')
        ts[cnt0 % 2].remove(cnt0)
        q = deque([cnt0])
        ans = 0
        while q:
            for _ in range(len(q)):
                cur = q.popleft()
                if cur == 0:
                    return ans
                l = cur + k - 2 * min(cur, k)
                r = cur + k - 2 * max(k - n + cur, 0)
                t = ts[l % 2]
                j = t.bisect_left(l)
                while j < len(t) and t[j] <= r:
                    q.append(t[j])
                    t.remove(t[j])
            ans += 1
        return -1

def _test():
    s = Solution()
    assert s.minOperations("110", 1) == 1
    assert s.minOperations("0101", 3) == 2
    assert s.minOperations("101", 2) == -1
    print("LC 3666 OK")


if __name__ == "__main__":
    _test()
