"""
LC 4109. Count Valid Suffix Sorts II — Hard
https://leetcode.com/problems/count-valid-suffix-sorts-ii/
Best: for each i, keep prefix + sort suffix descending; count matches in target windows
"""
from collections import Counter


class Solution:
    def countSuffixSorts(self, nums: list[int], target: list[int]) -> list[int]:
        n, m = len(nums), len(target)
        patterns: Counter = Counter()
        for i in range(n):
            arr = nums[:i] + sorted(nums[i:], reverse=True)
            patterns[tuple(arr)] += 1
        return [patterns[tuple(target[k : k + n])] for k in range(m - n + 1)]


def _test():
    s = Solution()
    assert s.countSuffixSorts([2, 2, 3], [2, 2, 3, 2, 1, 3, 2, 2]) == [1, 1, 0, 0, 0, 1]
    assert s.countSuffixSorts([5, 6, 5], [5, 6, 5, 5, 6, 5]) == [2, 1, 0, 2]
    print("LC 4109 OK")


if __name__ == "__main__":
    _test()
