"""
LC 2242. Maximum Score of a Node Sequence — Hard
https://leetcode.com/problems/maximum-score-of-a-node-sequence/
Best: keep top-3 neighbors + enum edges as middle — O(n + m)
"""

from typing import List


class Solution:
    def maximumScore(self, scores: List[int], edges: List[List[int]]) -> int:
        n = len(scores)
        top = [[] for _ in range(n)]  # up to 3 neighbors by score desc

        def add(u: int, v: int) -> None:
            top[u].append(v)
            top[u].sort(key=lambda x: scores[x], reverse=True)
            if len(top[u]) > 3:
                top[u].pop()

        for a, b in edges:
            add(a, b)
            add(b, a)

        ans = -1
        for b, c in edges:
            for a in top[b]:
                if a == c:
                    continue
                for d in top[c]:
                    if d == b or d == a:
                        continue
                    ans = max(ans, scores[a] + scores[b] + scores[c] + scores[d])
        return ans


def _test():
    s = Solution()
    assert s.maximumScore(
        [5, 2, 9, 8, 4], [[0, 1], [1, 2], [2, 3], [0, 2], [1, 3], [2, 4]]
    ) == 24
    assert s.maximumScore(
        [9, 20, 6, 4, 11, 12], [[0, 3], [5, 3], [2, 4], [1, 3]]
    ) == -1
    print("LC 2242 OK")


if __name__ == "__main__":
    _test()
