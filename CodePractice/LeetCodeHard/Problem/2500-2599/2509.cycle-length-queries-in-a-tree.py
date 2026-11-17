"""
LC 2509. Cycle Length Queries in a Tree — Hard
https://leetcode.com/problems/cycle-length-queries-in-a-tree/
Best: LCA on complete binary tree via bit ops — O(q log n)
"""

from typing import List


class Solution:
    def cycleLengthQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        ans = []
        for a, b in queries:
            x, y = a, b
            steps = 0
            while x != y:
                if x > y:
                    x //= 2
                else:
                    y //= 2
                steps += 1
            ans.append(steps + 1)
        return ans


def _test():
    s = Solution()
    assert s.cycleLengthQueries(3, [[5, 3], [4, 7], [2, 3]]) == [4, 5, 3]
    assert s.cycleLengthQueries(2, [[1, 2]]) == [2]
    print("LC 2509 OK")


if __name__ == "__main__":
    _test()
