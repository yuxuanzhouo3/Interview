"""
LC 220. Contains Duplicate III — Hard
Best: Bucket hash by t-width — O(n)
"""

from typing import List


class Solution:
    def containsNearbyAlmostDuplicate(self, nums: List[int], indexDiff: int, valueDiff: int) -> bool:
        if valueDiff < 0:
            return False
        buckets = {}
        w = valueDiff + 1

        def get_id(x):
            return x // w

        for i, x in enumerate(nums):
            bid = get_id(x)
            if bid in buckets:
                return True
            if bid - 1 in buckets and abs(x - buckets[bid - 1]) < w:
                return True
            if bid + 1 in buckets and abs(x - buckets[bid + 1]) < w:
                return True
            buckets[bid] = x
            if i >= indexDiff:
                del buckets[get_id(nums[i - indexDiff])]
        return False


def _test():
    s = Solution()
    assert s.containsNearbyAlmostDuplicate([1, 2, 3, 1], 3, 0) is True
    assert s.containsNearbyAlmostDuplicate([1, 5, 9, 1, 5, 9], 2, 3) is False
    print("LC 220 OK")


if __name__ == "__main__":
    _test()
