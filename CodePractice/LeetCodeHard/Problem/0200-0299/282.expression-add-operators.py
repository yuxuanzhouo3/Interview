"""
LC 282. Expression Add Operators — Hard
Best: DFS build expr with + - * tracking last operand
"""

from typing import List


class Solution:
    def addOperators(self, num: str, target: int) -> List[str]:
        ans = []

        def dfs(i, expr, value, last):
            if i == len(num):
                if value == target:
                    ans.append(expr)
                return
            for j in range(i, len(num)):
                if j > i and num[i] == "0":
                    break
                cur_s = num[i:j + 1]
                cur = int(cur_s)
                if i == 0:
                    dfs(j + 1, cur_s, cur, cur)
                else:
                    dfs(j + 1, expr + "+" + cur_s, value + cur, cur)
                    dfs(j + 1, expr + "-" + cur_s, value - cur, -cur)
                    dfs(j + 1, expr + "*" + cur_s, value - last + last * cur, last * cur)

        dfs(0, "", 0, 0)
        return ans


def _test():
    s = Solution()
    assert sorted(s.addOperators("123", 6)) == sorted(["1+2+3", "1*2*3"])
    assert sorted(s.addOperators("232", 8)) == sorted(["2*3+2", "2+3*2"])
    print("LC 282 OK")


if __name__ == "__main__":
    _test()
