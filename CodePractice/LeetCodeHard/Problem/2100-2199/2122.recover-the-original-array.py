"""
LC 2122. Recover the Original Array — Hard
https://leetcode.com/problems/recover-the-original-array/
Best: Enumerate k from sorted pairs + multiset check — O(n^2)
"""


from typing import List
from collections import Counter


class Solution:
    def recoverArray(self, nums: List[int]) -> List[int]:
        nums = sorted(nums)
        n = len(nums) // 2
        first = nums[0]
        for i in range(1, len(nums)):
            diff = nums[i] - first
            if diff == 0 or diff % 2:
                continue
            k = diff // 2
            cnt = Counter(nums)
            ans = []
            ok = True
            for x in nums:
                if cnt[x] == 0:
                    continue
                if cnt[x + 2 * k] == 0:
                    ok = False
                    break
                cnt[x] -= 1
                cnt[x + 2 * k] -= 1
                ans.append(x + k)
            if ok and len(ans) == n:
                return ans
        return []


def _test():
    s = Solution()
    assert sorted(s.recoverArray([2, 10, 6, 4, 8, 12])) in ([3, 7, 11], [5, 7, 9])
    assert sorted(s.recoverArray([1, 1, 3, 3])) == [2, 2]
    assert sorted(s.recoverArray([5, 435])) == [220]
    print("LC 2122 OK")


if __name__ == "__main__":
    _test()

