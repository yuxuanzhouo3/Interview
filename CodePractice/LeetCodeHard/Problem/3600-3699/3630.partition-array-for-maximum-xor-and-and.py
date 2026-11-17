"""
LC 3630. Partition Array for Maximum XOR and AND — Hard
https://leetcode.com/problems/partition-array-for-maximum-xor-and-and/
Tags: Array, Math, Greedy, Bit Manipulation, Enumeration
Best: 最优解
"""
class Solution(object):
    def maximizeXorAndXor(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        def max_xor_subset(nums):  # Time: O(nlogr)
            base = [0]*l 
            for x in nums:  # gaussian elimination over GF(2)
                for i in reversed(range(len(base))):
                    if not x&(1<<i):
                        continue
                    if base[i] == 0:
                        base[i] = x
                        break
                    x ^= base[i]
            max_xor = 0
            for b in reversed(base):  # greedy
                if (max_xor^b) > max_xor:
                    max_xor ^= b
            return max_xor

        l = max(nums).bit_length()
        n = len(nums)
        and_arr = [0]*(1<<n)
        xor_arr = [0]*(1<<n)
        for mask in range(1, 1<<n):
            lb = mask&-mask
            i = lb.bit_length()-1
            and_arr[mask] = and_arr[mask^lb]&nums[i] if mask^lb else nums[i]
            xor_arr[mask] = xor_arr[mask^lb]^nums[i]
        result = 0
        full_mask = (1<<n)-1
        for mask in range(1, 1<<n):
            total_and = and_arr[mask]
            total_xor = xor_arr[full_mask^mask]
            max_xor = max_xor_subset(((nums[i]&~total_xor) for i in range(n) if not (mask&(1<<i))))
            result = max(result, total_and+total_xor+2*max_xor)
        return result


# Time:  O(nlogr * 2^n)
# Space: O(1)
# bitmasks, greedy

def _test():
    s = Solution()
    assert s.maximizeXorAndXor(nums=[2, 3]) == 5
    assert s.maximizeXorAndXor(nums=[1, 3, 2]) == 6
    assert s.maximizeXorAndXor(nums=[2, 3, 6, 7]) == 15
    print("LC 3630 OK")

if __name__ == "__main__":
    _test()
