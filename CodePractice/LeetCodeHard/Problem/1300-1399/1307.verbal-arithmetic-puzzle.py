"""
LC 1307. Verbal Arithmetic Puzzle — Hard
https://leetcode.com/problems/verbal-arithmetic-puzzle/
Best: Column backtracking with leading-zero constraints — O(10!)
"""

from typing import List


class Solution:
    def isSolvable(self, words: List[str], result: str) -> bool:
        words = words + [result]
        max_len = max(len(w) for w in words)
        if len(result) < max_len:
            return False
        leading = {w[0] for w in words if len(w) > 1}
        chars = []
        seen = set()
        for col in range(max_len):
            for w in words:
                if col < len(w):
                    c = w[-(col + 1)]
                    if c not in seen:
                        seen.add(c)
                        chars.append(c)

        assign = {}
        used = [False] * 10

        def dfs(pos: int, carry: int) -> bool:
            if pos == max_len:
                return carry == 0
            # process one column: sum letters of addends - result digit
            # Actually process letter by letter at current column first
            # Find unassigned letter in this column
            col_chars = []
            for w in words[:-1]:
                if pos < len(w):
                    col_chars.append(w[-(pos + 1)])
            res_ch = result[-(pos + 1)] if pos < len(result) else None

            def solve_col(i: int, s: int) -> bool:
                if i == len(col_chars):
                    s += carry
                    digit = s % 10
                    new_carry = s // 10
                    if res_ch is None:
                        return digit == 0 and dfs(pos + 1, new_carry)
                    if res_ch in assign:
                        return assign[res_ch] == digit and dfs(pos + 1, new_carry)
                    if used[digit]:
                        return False
                    if digit == 0 and res_ch in leading:
                        return False
                    assign[res_ch] = digit
                    used[digit] = True
                    ok = dfs(pos + 1, new_carry)
                    del assign[res_ch]
                    used[digit] = False
                    return ok
                ch = col_chars[i]
                if ch in assign:
                    return solve_col(i + 1, s + assign[ch])
                for d in range(10):
                    if used[d]:
                        continue
                    if d == 0 and ch in leading:
                        continue
                    assign[ch] = d
                    used[d] = True
                    if solve_col(i + 1, s + d):
                        return True
                    del assign[ch]
                    used[d] = False
                return False

            return solve_col(0, 0)

        return dfs(0, 0)


def _test():
    s = Solution()
    assert s.isSolvable(["SEND", "MORE"], "MONEY") is True
    assert s.isSolvable(["SIX", "SEVEN", "SEVEN"], "TWENTY") is True
    assert s.isSolvable(["LEET", "CODE"], "POINT") is False
    print("LC 1307 OK")


if __name__ == "__main__":
    _test()
