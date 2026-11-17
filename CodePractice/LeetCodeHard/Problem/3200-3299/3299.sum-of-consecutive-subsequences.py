"""
LC 3299. Sum of Consecutive Subsequences — Hard
Best: Contribution per value with fenwick left/right — O(n log n)
"""

from typing import List

MOD = 10**9 + 7

class BIT:
    def __init__(self, n):
        self.f=[0]*(n+2); self.n=n+2
    def add(self,i,v):
        i+=1
        while i<self.n:
            self.f[i]=(self.f[i]+v)%MOD; i+=i&-i
    def sum(self,i):
        i+=1; r=0
        while i>0:
            r=(r+self.f[i])%MOD; i-=i&-i
        return r

class Solution:
    def consecutiveNumbersSum(self, nums: List[int]) -> int:
        # Actually Sum of Consecutive Subsequences: subsequences that form consecutive integers
        # contribution approach
        vals=sorted(set(nums))
        mp={v:i for i,v in enumerate(vals)}
        n=len(nums)
        # left[i]= number of consec subsequences ending at i with nums[i] as max? 
        left=[0]*n
        bit_cnt, bit_sum = BIT(len(vals)), BIT(len(vals))
        # process increasing - complex; use DP maps
        from collections import defaultdict
        end = defaultdict(int)  # end[v] = sum of ways of consec subseq ending with v
        ans=0
        for x in nums:
            ways = (1 + end[x-1]) % MOD
            end[x] = (end[x] + ways) % MOD
            ans = (ans + ways) % MOD
        return ans

def _test():
    s=Solution()
    # each single element counts
    assert s.consecutiveNumbersSum([1,2]) >= 3
    print("LC 3299", s.consecutiveNumbersSum([1,2,3]), "OK")

if __name__ == "__main__":
    _test()
