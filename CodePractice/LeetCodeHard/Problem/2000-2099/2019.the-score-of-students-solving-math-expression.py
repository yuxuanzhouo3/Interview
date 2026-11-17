"""
LC 2019. The Score of Students Solving Math Expression — Hard
https://leetcode.com/problems/the-score-of-students-solving-math-expression/
Best: Interval DP all * / + results + score — O(m³ · |S|²)
"""

from collections import Counter
from typing import List

class Solution:
    def scoreOfStudents(self, s: str, answers: List[int]) -> int:
        def cal(expr: str) -> int:
            res, pre = 0, int(expr[0])
            for i in range(1, n, 2):
                if expr[i] == "*":
                    pre *= int(expr[i + 1])
                else:
                    res += pre
                    pre = int(expr[i + 1])
            return res + pre

        n = len(s)
        x = cal(s)
        m = (n + 1) >> 1
        f = [[set() for _ in range(m)] for _ in range(m)]
        for i in range(m):
            f[i][i] = {int(s[i << 1])}
        for i in range(m - 1, -1, -1):
            for j in range(i, m):
                for k in range(i, j):
                    for l in f[i][k]:
                        for r in f[k + 1][j]:
                            if s[k << 1 | 1] == "+" and l + r <= 1000:
                                f[i][j].add(l + r)
                            elif s[k << 1 | 1] == "*" and l * r <= 1000:
                                f[i][j].add(l * r)
        cnt = Counter(answers)
        ans = cnt[x] * 5
        for k, v in cnt.items():
            if k != x and k in f[0][m - 1]:
                ans += v * 2
        return ans

def _test():

    s = Solution()
    assert s.scoreOfStudents("7+3*1*2", [20, 13, 42]) == 7
    assert s.scoreOfStudents("3+5*2", [13, 0, 10, 13, 13, 16, 16]) == 19
    assert s.scoreOfStudents("6+0*1", [12, 9, 6, 4, 8, 6]) == 10
    print("LC 2019 OK")

if __name__ == "__main__":
    _test()
