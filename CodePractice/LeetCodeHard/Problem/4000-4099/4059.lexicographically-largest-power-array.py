"""
LC 4059. Lexicographically Largest Power Array — Hard
https://leetcode.com/problems/lexicographically-largest-power-array/
Best: sort descending; for each bit measure longest prefix with that bit set
"""


class Solution:
    def largestPower(self, nums: list[int]) -> list[int]:
        nums = sorted(nums, reverse=True)
        n = len(nums)
        power = [0] * 15
        for i in range(15):
            bit = 14 - i
            j = 0
            while j < n and (nums[j] >> bit) & 1:
                j += 1
            power[i] = j
        return power


def _test():
    s = Solution()
    assert s.largestPower([7, 5]) == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 2]
    assert s.largestPower([3, 1, 7]) == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3]
    print("LC 4059 OK")


if __name__ == "__main__":
    _test()
