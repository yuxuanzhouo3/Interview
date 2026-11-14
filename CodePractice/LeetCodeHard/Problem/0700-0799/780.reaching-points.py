"""
LC 780. Reaching Points — Hard
Best: Reverse modulo from (tx,ty) — O(log)
"""

class Solution:
    def reachingPoints(self, sx: int, sy: int, tx: int, ty: int) -> bool:
        while tx >= sx and ty >= sy:
            if tx == sx and ty == sy:
                return True
            if tx > ty:
                if ty > sy:
                    tx %= ty
                else:
                    return (tx - sx) % ty == 0
            else:
                if tx > sx:
                    ty %= tx
                else:
                    return (ty - sy) % tx == 0
        return False


def _test():
    s = Solution()
    assert s.reachingPoints(1, 1, 3, 5) is True
    assert s.reachingPoints(1, 1, 2, 2) is False
    print("LC 780 OK")


if __name__ == "__main__":
    _test()
