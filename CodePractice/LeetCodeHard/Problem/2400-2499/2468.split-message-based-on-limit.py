"""
LC 2468. Split Message Based on Limit — Hard
https://leetcode.com/problems/split-message-based-on-limit/
Best: Enumerate part count by digit budget — O(n)
"""

from typing import List

class Solution:
    def splitMessage(self, message: str, limit: int) -> List[str]:
        n = len(message)
        sa = 0
        for k in range(1, n + 1):
            sa += len(str(k))
            sb = len(str(k)) * k
            sc = 3 * k
            if limit * k - (sa + sb + sc) >= n:
                ans = []
                i = 0
                for j in range(1, k + 1):
                    tail = f"<{j}/{k}>"
                    t = message[i : i + limit - len(tail)] + tail
                    ans.append(t)
                    i += limit - len(tail)
                return ans
        return []

def _test():

    s = Solution()
    r = s.splitMessage("this is really a very awesome message", 9)
    assert len(r) == 14 and r[0].endswith("<1/14>") and "".join(x[: x.index("<")] for x in r) == "this is really a very awesome message"
    r2 = s.splitMessage("short message", 15)
    assert r2 == ["short mess<1/2>", "age<2/2>"]
    print("LC 2468 OK")

if __name__ == "__main__":
    _test()
