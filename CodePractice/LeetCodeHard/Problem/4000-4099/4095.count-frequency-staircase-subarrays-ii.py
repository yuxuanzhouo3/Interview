"""
LC 4095. Count Frequency Staircase Subarrays II — Hard
https://leetcode.com/problems/count-frequency-staircase-subarrays-ii/
Best: expand from each left; staircase length must be k(k+1)/2
"""


class Solution:
    def countStaircase(self, nums: list[int]) -> int:
        n = len(nums)
        ans = n  # all singles
        # A staircase with k distinct has length = 1+2+...+k = k(k+1)/2
        # Enumerate left endpoint; greedily grow
        for i in range(n):
            freq = {}
            order = []  # first appearance order
            for j in range(i, n):
                x = nums[j]
                if x not in freq:
                    freq[x] = 0
                    order.append(x)
                freq[x] += 1
                k = len(order)
                need = k * (k + 1) // 2
                length = j - i + 1
                if length > need:
                    break  # too many extras somehow — actually can exceed then fail forever for this i? 
                # Check if frequencies match 1..k in order
                if length == need:
                    ok = True
                    for t, val in enumerate(order, 1):
                        if freq[val] != t:
                            ok = False
                            break
                    if ok and k >= 2:
                        ans += 1
                elif length > need:
                    break
        return ans

def _test():
    s = Solution()
    assert s.countStaircase(nums=[5, 8, 8, 2, 2, 2]) == 9
    assert s.countStaircase(nums=[6, 6, 9, 9]) == 5
    assert s.countStaircase(nums=[2, 3, 4, 4, 3, 4]) == 8
    print("LC 4095 OK")


if __name__ == "__main__":
    _test()
