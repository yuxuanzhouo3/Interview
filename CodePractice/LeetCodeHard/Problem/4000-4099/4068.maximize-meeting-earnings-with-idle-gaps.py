from math import inf
import bisect

class Solution:
    def maxEarnings(self, meetings: list[list[int]]) -> int:
        n = len(meetings)
        meetings.sort(key=lambda x: x[1])
        ends = [m[1] for m in meetings]
        pre_max = [-inf] * (n + 1)
        ans = 0
        for i, (start, end, revenue) in enumerate(meetings):
            val = revenue
            if start >= meetings[0][1]:
                j = bisect.bisect_right(ends, start, hi=i)
                val += pre_max[j] + start
            ans = max(ans, val)
            pre_max[i + 1] = max(pre_max[i], val - end)
        return ans

def _test():
    s = Solution()
    assert s.maxEarnings([[2,5,4],[6,8,3]]) == 8
    assert s.maxEarnings([[3,5,4],[4,7,8],[8,10,3]]) == 12
    assert s.maxEarnings([[1,2,2],[4,5,2],[7,9,3]]) == 11
    print("LC 4068 OK")


if __name__ == "__main__":
    _test()
