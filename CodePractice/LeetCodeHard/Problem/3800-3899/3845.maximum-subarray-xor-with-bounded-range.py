"""
LC 3845. Maximum Subarray XOR with Bounded Range — Hard
https://leetcode.com/problems/maximum-subarray-xor-with-bounded-range/
Best: doocs/leetcode reference solution (verified on examples)
"""

from collections import deque, defaultdict, Counter


class TrieNode:
    __slots__ = ("children", "count")

    def __init__(self):
        self.children = [None, None]
        self.count = 0


class Solution:
    def maxXor(self, nums: list[int], k: int) -> int:
        root = TrieNode()

        def update(value: int, delta: int) -> None:
            cur = root
            for bit in range(14, -1, -1):
                b = (value >> bit) & 1
                if cur.children[b] is None:
                    cur.children[b] = TrieNode()
                cur = cur.children[b]
                cur.count += delta

        def get_max_xor(value: int) -> int:
            cur = root
            ans = 0
            for bit in range(14, -1, -1):
                b = (value >> bit) & 1
                opp = 1 - b
                if cur.children[opp] is not None and cur.children[opp].count > 0:
                    ans |= 1 << bit
                    cur = cur.children[opp]
                else:
                    cur = cur.children[b]
            return ans

        n = len(nums)
        prefix = [0] * (n + 1)
        for i, x in enumerate(nums):
            prefix[i + 1] = prefix[i] ^ x

        maxq, minq = deque(), deque()
        left = 0
        ans = 0
        update(prefix[0], 1)
        for right, x in enumerate(nums):
            while maxq and nums[maxq[-1]] <= x:
                maxq.pop()
            while minq and nums[minq[-1]] >= x:
                minq.pop()
            maxq.append(right)
            minq.append(right)
            while nums[maxq[0]] - nums[minq[0]] > k:
                if maxq[0] == left:
                    maxq.popleft()
                if minq[0] == left:
                    minq.popleft()
                update(prefix[left], -1)
                left += 1
            ans = max(ans, get_max_xor(prefix[right + 1]))
            update(prefix[right + 1], 1)
        return ans

def _test():
    s = Solution()
    assert s.maxXor([5,4,5,6], 2) == 7
    assert s.maxXor([5,4,5,6], 1) == 6
    print("LC 3845 OK")


if __name__ == "__main__":
    _test()
