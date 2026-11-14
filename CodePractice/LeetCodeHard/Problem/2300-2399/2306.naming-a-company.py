"""
LC 2306. Naming a Company — Hard
https://leetcode.com/problems/naming-a-company/
Best: Group by first letter suffix sets — O(n L + 26^2)
"""

from typing import List


class Solution:
    def distinctNames(self, ideas: List[str]) -> int:
        groups = [set() for _ in range(26)]
        for w in ideas:
            groups[ord(w[0]) - 97].add(w[1:])
        ans = 0
        for i in range(26):
            for j in range(i + 1, 26):
                inter = len(groups[i] & groups[j])
                ans += (len(groups[i]) - inter) * (len(groups[j]) - inter) * 2
        return ans


def _test():
    s = Solution()
    assert s.distinctNames(["coffee", "donuts", "time", "toffee"]) == 6
    assert s.distinctNames(["lack", "back"]) == 0
    print("LC 2306 OK")


if __name__ == "__main__":
    _test()
