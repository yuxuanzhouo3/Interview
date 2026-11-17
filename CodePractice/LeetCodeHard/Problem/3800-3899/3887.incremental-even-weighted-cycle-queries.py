"""
LC 3887. Incremental Even Weighted Cycle Queries — Hard
https://leetcode.com/problems/incremental-even-weighted-cycle-queries/
Best: parity Union-Find (XOR to root); reject edges that create odd-weight cycles
"""


class Solution:
    def numberOfEdgesAdded(self, n: int, edges: list[list[int]]) -> int:
        parent = list(range(n))
        xor = [0] * n

        def find(x: int) -> int:
            if parent[x] != x:
                p = find(parent[x])
                xor[x] ^= xor[parent[x]]
                parent[x] = p
            return parent[x]

        ans = 0
        for u, v, w in edges:
            pu, pv = find(u), find(v)
            if pu == pv:
                if (xor[u] ^ xor[v]) == w:
                    ans += 1
            else:
                parent[pu] = pv
                xor[pu] = xor[u] ^ xor[v] ^ w
                ans += 1
        return ans


def _test():
    s = Solution()
    assert s.numberOfEdgesAdded(3, [[0, 1, 1], [1, 2, 1], [0, 2, 1]]) == 2
    assert s.numberOfEdgesAdded(3, [[0, 1, 1], [1, 2, 1], [0, 2, 0]]) == 3
    print("LC 3887 OK")


if __name__ == "__main__":
    _test()
