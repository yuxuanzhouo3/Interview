"""
LC 3267. Count Almost Equal Pairs II — Hard
https://leetcode.com/problems/count-almost-equal-pairs-ii/
Best: 方法一：排序 + 枚举
"""

from collections import defaultdict

class Solution:
    def countPairs(self, nums: list[int]) -> int:
        nums.sort()
        ans = 0
        cnt = defaultdict(int)
        for x in nums:
            vis = {x}
            s = list(str(x))
            m = len(s)
            for j in range(m):
                for i in range(j):
                    s[i], s[j] = s[j], s[i]
                    vis.add(int("".join(s)))
                    for q in range(i + 1, m):
                        for p in range(i + 1, q):
                            s[p], s[q] = s[q], s[p]
                            vis.add(int("".join(s)))
                            s[p], s[q] = s[q], s[p]
                    s[i], s[j] = s[j], s[i]
            ans += sum(cnt[x] for x in vis)
            cnt[x] += 1
        return ans

def _test():
    s = Solution()
    assert s.countPairs(nums=[1023, 2310, 2130, 213]) == 4
    assert s.countPairs(nums=[1, 10, 100]) == 3
    print("LC 3267 OK")

if __name__ == "__main__":
    _test()
