"""
LC 3261. Count Substrings That Satisfy K-Constraint II — Hard
https://leetcode.com/problems/count-substrings-that-satisfy-k-constraint-ii/
Best: 方法一：滑动窗口 + 前缀和
"""

class Solution:
    def countKConstraintSubstrings(
        self, s: str, k: int, queries: list[list[int]]
    ) -> list[int]:
        cnt = [0, 0]
        i, n = 0, len(s)
        d = [n] * n
        pre = [0] * (n + 1)
        for j, x in enumerate(map(int, s)):
            cnt[x] += 1
            while cnt[0] > k and cnt[1] > k:
                d[i] = j
                cnt[int(s[i])] -= 1
                i += 1
            pre[j + 1] = pre[j] + j - i + 1
        ans = []
        for l, r in queries:
            p = min(r + 1, d[l])
            a = (1 + p - l) * (p - l) // 2
            b = pre[r + 1] - pre[p]
            ans.append(a + b)
        return ans

def _test():
    s = Solution()
    assert s.countKConstraintSubstrings(s='0001111', k=2, queries=[[0, 6]]) == [26]
    assert s.countKConstraintSubstrings(s='010101', k=1, queries=[[0, 5], [1, 4], [2, 3]]) == [15, 9, 3]
    print("LC 3261 OK")

if __name__ == "__main__":
    _test()
