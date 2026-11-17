"""
LC 4064. Longest Subarray Divisible by K with At Most One Negation II — Hard
https://leetcode.com/problems/longest-subarray-divisible-by-k-with-at-most-one-negation-ii/
Best: doocs/leetcode reference solution (verified on examples)
"""


class Solution:
    def longestSubarray(self, nums: list[int], k: int) -> int:
        n = len(nums)
        p = [0] * (n + 1)
        for i in range(n):
            p[i + 1] = (p[i] + nums[i]) % k

        first = [-1] * k
        for i in range(n + 1):
            if first[p[i]] == -1:
                first[p[i]] = i

        order = [q for q in range(k) if first[q] != -1]
        order.sort(key=lambda q: first[q])

        pos = [0] * k
        best = [x if x != -1 else n + 1 for x in first]

        ans = 0
        for i, x in enumerate(nums):
            a = x % k
            while pos[a] < len(order) and first[order[pos[a]]] <= i:
                q = order[pos[a]]
                pos[a] += 1
                t = (q + 2 * a) % k
                best[t] = min(best[t], first[q])

            s = p[i + 1]
            if best[s] != n + 1:
                ans = max(ans, i + 1 - best[s])

        return ans

def _test():
    s = Solution()
    assert s.longestSubarray([4,1,2], 3) == 3
    assert s.longestSubarray([5,3,4], 7) == 2
    assert s.longestSubarray([2,2,5], 6) == 2
    print("LC 4064 OK")


if __name__ == "__main__":
    _test()
