"""
LC 4047. Minimum Operations to Make XOR of All Elements Zero — Hard
https://leetcode.com/problems/minimum-operations-to-make-xor-of-all-elements-zero/
Best: kamyu104 reference solution (verified on examples)
"""

# Time:  O(n + r * d), r = max(nums), d = len(set(nums))
# Space: O(r + d)

# bi-bfs
class Solution(object):
    def minOperations(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        def bi_bfs(start, target):
            left, right = {start}, {target}
            lookup = set()
            steps = 0
            while left:
                if len(left) > len(right):
                    left, right = right, left
                for x in left:
                    lookup.add(x)
                new_left = set()
                for x in left:
                    if x in right:
                        return steps
                    for v in nums_set:
                        nx = x^v
                        if nx in lookup:
                            continue
                        new_left.add(nx)
                left = new_left
                steps += 1
            return -1

        total = 0
        for x in nums:
            total ^= x
        if total == 0:
            return 0
        nums_set = set(nums)
        if len(nums_set) == 1:
            return -1
        steps = bi_bfs(0, total)
        return steps if steps != len(nums) else -1


# Time:  O(n + r * d), r = max(nums), d = len(set(nums))
# Space: O(r + d)
# bfs

def _test():
    s = Solution()
    assert s.minOperations(nums=[8, 1, 4, 8, 2]) == 3
    assert s.minOperations(nums=[1, 2, 3]) == 0
    assert s.minOperations(nums=[1, 2, 4]) == -1
    print("LC 4047 OK")


if __name__ == "__main__":
    _test()
