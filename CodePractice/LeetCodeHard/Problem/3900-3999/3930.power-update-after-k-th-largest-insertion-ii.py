"""
LC 3930. Power Update After K-th Largest Insertion II — Hard
https://leetcode.com/problems/power-update-after-k-th-largest-insertion-ii/
Best: kamyu104 reference solution (verified on examples)
"""

# Time:  O((n + q) * log(n * q) + q * logr)
# Space: O(n + q)

from sortedcontainers import SortedList


# sorted list, fast exponentiation
class Solution(object):
    def powerUpdate(self, nums, p, queries):
        """
        :type nums: List[int]
        :type p: int
        :type queries: List[List[int]]
        :rtype: List[int]
        """
        MOD = 10**9+7
        sl = SortedList(nums)
        result = []
        for x, k in queries:
            sl.add(x)
            p = pow(p, sl[-k], MOD)
            result.append(p)
        return result


# Time:  O((n + q) * log(n * q) + q * logr)
# Space: O(n + q)
# sort, coordinate compression, fenwick tree, fast exponentiation
class BIT(object):  # 0-indexed.
    def __init__(self, n):
        self.__bit = [0]*(n+1)  # Extra one for dummy node.

    def add(self, i, val):
        i += 1  # Extra one for dummy node.
        while i < len(self.__bit):
            self.__bit[i] += val
            i += (i & -i)

    def query(self, i):
        i += 1  # Extra one for dummy node.
        ret = 0
        while i > 0:
            ret += self.__bit[i]
            i -= (i & -i)
        return ret
   
    def kth_element(self, k):
        floor_log2_n = (len(self.__bit)-1).bit_length()-1
        pow_i = 2**floor_log2_n
        total = pos = 0  # 1-indexed
        for _ in reversed(range(floor_log2_n+1)):  # O(logN)
            if pos+pow_i < len(self.__bit) and not total+self.__bit[pos+pow_i] >= k:
                total += self.__bit[pos+pow_i]
                pos += pow_i
            pow_i >>= 1
        return (pos+1)-1

def _test():
    s = Solution()
    assert s.powerUpdate(nums=[2], p=4, queries=[[3, 1], [1, 2]]) == [64, 4096]
    assert s.powerUpdate(nums=[7, 5], p=6, queries=[[4, 3], [7, 2]]) == [1296, 220296870]
    print("LC 3930 OK")


if __name__ == "__main__":
    _test()
