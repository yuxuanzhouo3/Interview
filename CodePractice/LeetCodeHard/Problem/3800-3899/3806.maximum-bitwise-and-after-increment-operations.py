"""
LC 3806. Maximum Bitwise AND After Increment Operations — Hard
https://leetcode.com/problems/maximum-bitwise-and-after-increment-operations/
Best: doocs/leetcode reference solution (verified on examples)
"""


class Solution:
    def maximumAND(self, nums: list[int], k: int, m: int) -> int:
        mx = (max(nums) + k).bit_length()
        ans = 0
        cost = [0] * len(nums)
        for bit in range(mx - 1, -1, -1):
            target = ans | (1 << bit)
            for i, x in enumerate(nums):
                j = (target & ~x).bit_length()
                mask = (1 << j) - 1
                cost[i] = (target & mask) - (x & mask)
            cost.sort()
            if sum(cost[:m]) <= k:
                ans = target
        return ans

def _test():
    s = Solution()
    assert s.maximumAND([3,1,2], 8, 2) == 6
    assert s.maximumAND([1,2,8,4], 7, 3) == 4
    assert s.maximumAND([1,1], 3, 2) == 2
    print("LC 3806 OK")


if __name__ == "__main__":
    _test()
