"""
LC 2459. Sort Array By Moving Items to Empty Space — Hard
Best: Cycle decomposition with 0 — O(n)
"""

from typing import List


class Solution:
    def sortArray(self, nums: List[int]) -> int:
        def solve(target):
            # target[i] should be at i; 0 empty
            pos = {v: i for i, v in enumerate(nums)}
            arr = nums[:]
            ans = 0
            for i in range(len(arr)):
                while arr[i] != target[i]:
                    # swap empty with what belongs at empty's position in target? 
                    j = pos[0]
                    # want target[j] to come to j
                    need = target[j]
                    k = pos[need]
                    # swap 0 and need
                    arr[j], arr[k] = arr[k], arr[j]
                    pos[0], pos[need] = k, j
                    ans += 1
            return ans

        n = len(nums)
        # two targets: [0,1,2,...,n-1] or [1,2,...,n-1,0]
        a = solve(list(range(n)))
        b = solve(list(range(1, n)) + [0])
        return min(a, b)


def _test():
    s = Solution()
    assert s.sortArray([4, 2, 0, 3, 1]) == 3
    assert s.sortArray([1, 2, 3, 4, 0]) == 0
    print("LC 2459 OK")


if __name__ == "__main__":
    _test()
