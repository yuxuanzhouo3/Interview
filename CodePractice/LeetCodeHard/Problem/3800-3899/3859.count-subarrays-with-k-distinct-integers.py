"""
LC 3859. Count Subarrays With K Distinct Integers — Hard
https://leetcode.com/problems/count-subarrays-with-k-distinct-integers/
Best: doocs/leetcode reference solution (verified on examples)
"""

from collections import defaultdict, Counter


class Solution:
    def countSubarrays(self, nums: list[int], k: int, m: int) -> int:
        def f(lim: int) -> int:
            cnt = Counter()
            t = 0
            ans = l = 0
            for x in nums:
                cnt[x] += 1
                if cnt[x] == m:
                    t += 1
                while len(cnt) >= lim and t >= k:
                    y = nums[l]
                    cnt[y] -= 1
                    if cnt[y] == m - 1:
                        t -= 1
                    if cnt[y] == 0:
                        cnt.pop(y)
                    l += 1
                ans += l
            return ans

        return f(k) - f(k + 1)

def _test():
    s = Solution()
    assert s.countSubarrays([1,2,1,2,2], 2, 2) == 2
    assert s.countSubarrays([3,1,2,4], 2, 1) == 3
    print("LC 3859 OK")


if __name__ == "__main__":
    _test()
