"""
LC 3181. Maximum Total Reward Using Operations II — Hard
https://leetcode.com/problems/maximum-total-reward-using-operations-ii/
Best: Bitset DP — O(n * max_reward / word)
"""

from typing import List


class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        rewardValues = sorted(set(rewardValues))
        # bit i set means sum i achievable; can take x if current sum < x
        bits = 1  # sum 0
        for x in rewardValues:
            # take lower bits < x, shift by x
            mask = (1 << x) - 1
            bits |= (bits & mask) << x
        return bits.bit_length() - 1


def _test():
    s = Solution()
    assert s.maxTotalReward([1, 1, 3, 3]) == 4
    assert s.maxTotalReward([1, 6, 4, 3, 2]) == 11
    print("LC 3181 OK")


if __name__ == "__main__":
    _test()
