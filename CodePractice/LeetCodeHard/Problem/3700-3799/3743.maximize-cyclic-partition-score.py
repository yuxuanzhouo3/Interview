"""
LC 3743. Maximize Cyclic Partition Score — Hard
https://leetcode.com/problems/maximize-cyclic-partition-score/
Tags: Array, Dynamic Programming
Best: 最优解
"""
# Time:  O(n * k)
# Space: O(k)

# dp
class Solution:
    def maximumScore(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        def best_time_to_buy_and_sell_stock_v(base):
            dp = [0]*(len(nums)+1)
            result = 0
            for i in range(k):
                x, y = float("-inf"), float("-inf")
                new_dp = [float("-inf")]*(len(nums)+1)
                for j in range(i, len(nums)):
                    x, y = max(x, dp[j]-nums[(base+j)%len(nums)]), max(y, dp[j]+nums[(base+j)%len(nums)])
                    new_dp[j+1] = max(new_dp[j], x+nums[(base+j)%len(nums)], y-nums[(base+j)%len(nums)])
                dp = new_dp
                result = max(result, dp[-1])
            return result
    
        i = min(range(len(nums)), key=lambda x: nums[x])
        return max(best_time_to_buy_and_sell_stock_v(i), best_time_to_buy_and_sell_stock_v(i+1))


def _test():
    s = Solution()
    assert s.maximumScore(nums=[1, 3, 4], k=2) == 3
    assert s.maximumScore(nums=[1, 1000000000, 1000000000], k=3) == 999999999
    assert s.maximumScore(nums=[1], k=1) == 0
    print("LC 3743 OK")

if __name__ == "__main__":
    _test()
