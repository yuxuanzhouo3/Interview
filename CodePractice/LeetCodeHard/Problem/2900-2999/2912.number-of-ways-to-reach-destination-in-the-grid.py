"""
LC 2912. Number of Ways to Reach Destination in the Grid — Hard
https://leetcode.com/problems/number-of-ways-to-reach-destination-in-the-grid/
Best: Combinatorics on row/col moves — O(1)/O(log MOD)
"""

class Solution:
    def numberOfWays(self, n: int, m: int, k: int, source: list, dest: list) -> int:
        MOD = 10**9 + 7
        # dp on (same_row, same_col) after steps
        # 4 states: (sr,sc): both same / row same col diff / row diff col same / both diff
        sx, sy = source
        dx, dy = dest
        # After t steps of moving to adjacent cell (4-dir), count ways to be at dest.
        # State machine:
        # A: at dest
        # B: same row as dest, wrong col
        # C: same col as dest, wrong row
        # D: neither
        def start_state():
            if sx == dx and sy == dy:
                return 0
            if sx == dx:
                return 1
            if sy == dy:
                return 2
            return 3

        A = B = C = D = 0
        st = start_state()
        if st == 0:
            A = 1
        elif st == 1:
            B = 1
        elif st == 2:
            C = 1
        else:
            D = 1

        for _ in range(k):
            nA = (B * 1 + C * 1) % MOD  # from same-row move into col, or same-col move into row — actually 2 dirs?
            # Correct transitions on grid n x m:
            # From A (at dest): can go to B (m-1 cols) or C (n-1 rows)
            # From B (same row, wrong col): to A (1), to B (m-2), to D (n-1)
            # From C: to A (1), to C (n-2), to D (m-1)
            # From D: to B (1 if move to dest row), to C (1 if move to dest col), to D (rest)
            nnA = (B + C) % MOD
            nnB = (A * (m - 1) + B * (m - 2) + D * 1) % MOD
            nnC = (A * (n - 1) + C * (n - 2) + D * 1) % MOD
            nnD = (B * (n - 1) + C * (m - 1) + D * (n + m - 4)) % MOD
            A, B, C, D = nnA, nnB, nnC, nnD
        return A


def _test():
    s = Solution()
    assert s.numberOfWays(3, 2, 2, [1, 1], [2, 2]) == 2
    assert s.numberOfWays(3, 4, 3, [1, 2], [2, 3]) == 9
    assert s.numberOfWays(3, 2, 0, [1, 1], [1, 1]) == 1
    print("LC 2912 OK")


if __name__ == "__main__":
    _test()

