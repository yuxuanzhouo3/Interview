"""
LC 3480. Maximize Subarrays After Removing One Conflicting Pair — Hard
https://leetcode.com/problems/maximize-subarrays-after-removing-one-conflicting-pair/
Best: 方法一：枚举 + 维护最小与次小值
"""

class Solution:
    def maxSubarrays(self, n: int, conflictingPairs: list[list[int]]) -> int:
        g = [[] for _ in range(n + 1)]
        for a, b in conflictingPairs:
            if a > b:
                a, b = b, a
            g[a].append(b)
        cnt = [0] * (n + 2)
        ans = add = 0
        b1 = b2 = n + 1
        for a in range(n, 0, -1):
            for b in g[a]:
                if b < b1:
                    b2, b1 = b1, b
                elif b < b2:
                    b2 = b
            ans += b1 - a
            cnt[b1] += b2 - b1
            add = max(add, cnt[b1])
        ans += add
        return ans

def _test():
    s = Solution()
    assert s.maxSubarrays(n=4, conflictingPairs=[[2, 3], [1, 4]]) == 9
    assert s.maxSubarrays(n=5, conflictingPairs=[[1, 2], [2, 5], [3, 5]]) == 12
    print("LC 3480 OK")

if __name__ == "__main__":
    _test()
