"""
LC 4086. Minimum Interval Selection to Cover Target Intervals — Hard
https://leetcode.com/problems/minimum-interval-selection-to-cover-target-intervals/
Best: doocs/leetcode reference solution (verified on examples)
"""


class Solution:
    def minSelection(
        self, availableIntervals: list[list[int]], targetIntervals: list[list[int]]
    ) -> int:
        availableIntervals.sort(key=lambda x: x[0])
        targetIntervals.sort(key=lambda x: x[0])
        ans = cur = 0
        i = 0
        for l, r in targetIntervals:
            while cur < r:
                uncover = max(l, cur)
                nxt = cur
                while (
                    i < len(availableIntervals) and availableIntervals[i][0] <= uncover
                ):
                    nxt = max(nxt, availableIntervals[i][1])
                    i += 1
                if nxt < uncover or nxt == cur:
                    return -1
                cur = nxt
                ans += 1
        return ans

def _test():
    s = Solution()
    assert s.minSelection([[1,4],[1,7],[3,9],[5,10],[10,15]], [[1,3],[2,5],[7,8],[10,13],[11,15],[12,13]]) == 3
    assert s.minSelection([[4,6],[1,4],[2,5]], [[3,4],[2,5],[6,7]]) == -1
    assert s.minSelection([[1,2],[3,4]], [[1,4]]) == -1
    print("LC 4086 OK")


if __name__ == "__main__":
    _test()
