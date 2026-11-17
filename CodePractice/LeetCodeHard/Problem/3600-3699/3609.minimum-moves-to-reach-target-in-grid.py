"""
LC 3609. Minimum Moves to Reach Target in Grid — Hard
https://leetcode.com/problems/minimum-moves-to-reach-target-in-grid/
Best: Reverse simulation — O(log(max(tx, ty)))
"""


class Solution:
    def minMoves(self, sx: int, sy: int, tx: int, ty: int) -> int:
        ans = 0
        while (tx, ty) != (sx, sy):
            if tx < sx or ty < sy:
                return -1
            if tx == ty:
                if sx == 0 and sy == ty:
                    return ans + 1
                if sy == 0 and sx == tx:
                    return ans + 1
                return -1
            if tx > ty:
                if ty == sy:
                    cur = tx
                    while cur > sx:
                        if cur > 2 * sy:
                            if cur % 2:
                                return -1
                            cur //= 2
                            ans += 1
                        elif cur == 2 * sy:
                            cur = sy  # doubling from x == sy
                            ans += 1
                        else:
                            # sy < cur < 2*sy (or cur <= sy handled below)
                            if cur > sy:
                                nxt = cur - sy
                                if nxt < sy:
                                    cur = nxt
                                    ans += 1
                                else:
                                    return -1
                            else:
                                nxt = cur - sy
                                if 0 <= nxt < sy:
                                    cur = nxt
                                    ans += 1
                                else:
                                    return -1
                    return ans if cur == sx else -1
                if tx > 2 * ty:
                    if tx % 2:
                        return -1
                    tx //= 2
                elif tx == 2 * ty:
                    tx = ty
                else:
                    x = tx - ty
                    if 0 <= x < ty:
                        tx = x
                    else:
                        return -1
                ans += 1
            else:
                if tx == sx:
                    cur = ty
                    while cur > sy:
                        if cur > 2 * sx:
                            if cur % 2:
                                return -1
                            cur //= 2
                            ans += 1
                        elif cur == 2 * sx:
                            cur = sx
                            ans += 1
                        else:
                            if cur > sx:
                                nxt = cur - sx
                                if nxt < sx:
                                    cur = nxt
                                    ans += 1
                                else:
                                    return -1
                            else:
                                nxt = cur - sx
                                if 0 <= nxt < sx:
                                    cur = nxt
                                    ans += 1
                                else:
                                    return -1
                    return ans if cur == sy else -1
                if ty > 2 * tx:
                    if ty % 2:
                        return -1
                    ty //= 2
                elif ty == 2 * tx:
                    ty = tx
                else:
                    y = ty - tx
                    if 0 <= y < tx:
                        ty = y
                    else:
                        return -1
                ans += 1
        return ans


def _test():
    s = Solution()
    assert s.minMoves(1, 2, 5, 4) == 2
    assert s.minMoves(0, 1, 2, 3) == 3
    assert s.minMoves(1, 1, 2, 2) == -1
    print("LC 3609 OK")


if __name__ == "__main__":
    _test()
