"""
LC 2158. Amount of New Area Painted Each Day — Hard
https://leetcode.com/problems/amount-of-new-area-painted-each-day/
Best: 跳跃数组 / 并查集找下一未涂点 — O(m + n)
"""

from typing import List


class Solution:
    def amountPainted(self, paint: List[List[int]]) -> List[int]:
        max_pos = max(e for _, e in paint)
        next_unpainted = list(range(max_pos + 2))
        ans = []

        def find(x: int) -> int:
            while next_unpainted[x] != x:
                next_unpainted[x] = next_unpainted[next_unpainted[x]]
                x = next_unpainted[x]
            return x

        for start, end in paint:
            painted = 0
            i = find(start)
            while i < end:
                painted += 1
                next_unpainted[i] = find(i + 1)
                i = next_unpainted[i]
            ans.append(painted)
        return ans


def _test():
    s = Solution()
    assert s.amountPainted([[1, 4], [4, 7], [5, 8]]) == [3, 3, 1]
    assert s.amountPainted([[1, 4], [5, 8], [4, 7]]) == [3, 3, 1]
    assert s.amountPainted([[1, 5], [2, 4]]) == [4, 0]
    print("LC 2158 OK")


if __name__ == "__main__":
    _test()
