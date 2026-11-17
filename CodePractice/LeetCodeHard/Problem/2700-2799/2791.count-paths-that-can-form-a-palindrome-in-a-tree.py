"""
LC 2791. Count Paths That Can Form a Palindrome in a Tree — Hard
https://leetcode.com/problems/count-paths-that-can-form-a-palindrome-in-a-tree/
Best: XOR path mask + meet-in-tree Counter — O(n · 26)
"""

from collections import defaultdict, Counter
from typing import List

class Solution:
    def countPalindromePaths(self, parent: List[int], s: str) -> int:
        def dfs(i: int, xor: int):
            nonlocal ans
            for j, v in g[i]:
                x = xor ^ v
                ans += cnt[x]
                for k in range(26):
                    ans += cnt[x ^ (1 << k)]
                cnt[x] += 1
                dfs(j, x)

        n = len(parent)
        g = defaultdict(list)
        for i in range(1, n):
            p = parent[i]
            g[p].append((i, 1 << (ord(s[i]) - ord("a"))))
        ans = 0
        cnt = Counter({0: 1})
        dfs(0, 0)
        return ans

def _test():

    s = Solution()
    assert s.countPalindromePaths([-1, 0, 0, 1, 1, 2], "acaabc") == 8
    assert s.countPalindromePaths([-1, 0, 0, 0, 0], "aaaaa") == 10
    print("LC 2791 OK")

if __name__ == "__main__":
    _test()
