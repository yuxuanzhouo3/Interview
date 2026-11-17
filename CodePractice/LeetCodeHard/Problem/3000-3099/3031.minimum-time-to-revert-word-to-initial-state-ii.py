"""
LC 3031. Minimum Time to Revert Word to Initial State II — Hard
https://leetcode.com/problems/minimum-time-to-revert-word-to-initial-state-ii/
Best: Z-algorithm / KMP border — O(n)
"""

class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        # z-box
        z = [0] * n
        l = r = 0
        for i in range(1, n):
            if i <= r:
                z[i] = min(r - i + 1, z[i - l])
            while i + z[i] < n and word[z[i]] == word[i + z[i]]:
                z[i] += 1
            if i + z[i] - 1 > r:
                l, r = i, i + z[i] - 1
        for t in range(1, (n + k - 1) // k + 1):
            start = t * k
            if start >= n:
                return t
            if z[start] >= n - start:
                return t
        return (n + k - 1) // k


def _test():
    s = Solution()
    assert s.minimumTimeToInitialState("abacaba", 3) == 2
    assert s.minimumTimeToInitialState("abacaba", 4) == 1
    assert s.minimumTimeToInitialState("abcbabcd", 2) == 4
    print("LC 3031 OK")


if __name__ == "__main__":
    _test()

