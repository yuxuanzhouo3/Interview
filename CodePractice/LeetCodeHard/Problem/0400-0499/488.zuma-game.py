"""
LC 488. Zuma Game — Hard
https://leetcode.com/problems/zuma-game/
Best: DFS + memo + group collapse — O(states)
"""

from functools import lru_cache
from collections import Counter


class Solution:
    def findMinStep(self, board: str, hand: str) -> int:
        def collapse(s: str) -> str:
            changed = True
            cur = s
            while changed:
                changed = False
                st = []
                for c in cur:
                    if st and st[-1][0] == c:
                        st[-1][1] += 1
                    else:
                        st.append([c, 1])
                nxt = []
                for ch, cnt in st:
                    if cnt < 3:
                        nxt.append(ch * cnt)
                    else:
                        changed = True
                cur = "".join(nxt)
            return cur

        hand_cnt = Counter(hand)

        @lru_cache(None)
        def dfs(b: str, ht: tuple) -> int:
            if not b:
                return 0
            h = Counter(dict(ht))
            ans = float("inf")
            i = 0
            n = len(b)
            while i < n:
                j = i
                while j < n and b[j] == b[i]:
                    j += 1
                color = b[i]
                need = max(0, 3 - (j - i))
                if need > 0 and h[color] >= need:
                    nb = collapse(b[:i] + b[j:])
                    h[color] -= need
                    nt = tuple(sorted((k, v) for k, v in h.items() if v > 0))
                    sub = dfs(nb, nt)
                    if sub >= 0:
                        ans = min(ans, sub + need)
                    h[color] += need
                i = j
            return -1 if ans == float("inf") else int(ans)

        ht0 = tuple(sorted((k, v) for k, v in hand_cnt.items() if v > 0))
        return dfs(collapse(board), ht0)


def _test():
    s = Solution()
    assert s.findMinStep("WRRBBW", "RB") == -1
    assert s.findMinStep("WWRRBBWW", "WRBRW") == 2
    assert s.findMinStep("G", "GGGGG") == 2
    assert s.findMinStep("RBYYBBRRB", "YRBGB") == 3
    print("LC 488 OK")


if __name__ == "__main__":
    _test()
