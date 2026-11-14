"""
LC 305. Number of Islands II — Hard
https://leetcode.com/problems/number-of-islands-ii/
Best: 动态加陆 + Union-Find — O(k α(mn))
"""

from typing import List


class Solution:
    def numIslands2(self, m: int, n: int, positions: List[List[int]]) -> List[int]:
        parent = [-1] * (m * n)
        rank = [0] * (m * n)
        count = 0
        ans = []

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            nonlocal count
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1
            count -= 1

        dirs = ((0, 1), (0, -1), (1, 0), (-1, 0))
        for r, c in positions:
            idx = r * n + c
            if parent[idx] != -1:
                ans.append(count)
                continue
            parent[idx] = idx
            count += 1
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n:
                    nidx = nr * n + nc
                    if parent[nidx] != -1:
                        union(idx, nidx)
            ans.append(count)
        return ans


def _test():
    s = Solution()
    assert s.numIslands2(3, 3, [[0, 0], [0, 1], [1, 2], [2, 1]]) == [1, 1, 2, 3]
    assert s.numIslands2(1, 1, [[0, 0]]) == [1]
    # duplicate position
    assert s.numIslands2(3, 3, [[0, 0], [0, 1], [0, 0]]) == [1, 1, 1]
    print("LC 305 OK")


if __name__ == "__main__":
    _test()
