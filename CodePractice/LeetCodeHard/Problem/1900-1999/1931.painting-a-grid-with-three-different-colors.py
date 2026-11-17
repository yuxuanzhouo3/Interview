"""
LC 1931. Painting a Grid With Three Different Colors — Hard
https://leetcode.com/problems/painting-a-grid-with-three-different-colors/
Best: Column state DP 3-coloring — O(3^m * m * n)
"""


class Solution:
    def colorTheGrid(self, m: int, n: int) -> int:
        MOD = 10**9 + 7
        # generate valid column colorings (base-3, colors 0,1,2)
        states = []

        def dfs(pos, prev_color, mask):
            if pos == m:
                states.append(mask)
                return
            for c in range(3):
                if c != prev_color:
                    dfs(pos + 1, c, mask * 3 + c)

        dfs(0, -1, 0)
        # compatibility
        compat = {a: [] for a in states}

        def colors(mask):
            out = []
            for _ in range(m):
                out.append(mask % 3)
                mask //= 3
            return out[::-1]

        cols = {s: colors(s) for s in states}
        for a in states:
            ca = cols[a]
            for b in states:
                cb = cols[b]
                if all(x != y for x, y in zip(ca, cb)):
                    compat[a].append(b)

        from collections import Counter
        dp = Counter(states)  # each state once for first column... wait each valid once
        dp = {s: 1 for s in states}
        for _ in range(n - 1):
            ndp = {s: 0 for s in states}
            for a, ways in dp.items():
                for b in compat[a]:
                    ndp[b] = (ndp[b] + ways) % MOD
            dp = ndp
        return sum(dp.values()) % MOD


def _test():
    s = Solution()
    assert s.colorTheGrid(1, 1) == 3
    assert s.colorTheGrid(1, 2) == 6
    assert s.colorTheGrid(5, 5) == 580986
    print("LC 1931 OK")


if __name__ == "__main__":
    _test()

