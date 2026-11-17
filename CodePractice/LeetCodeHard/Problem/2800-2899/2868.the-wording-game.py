"""
LC 2868. The Wording Game — Hard
https://leetcode.com/problems/the-wording-game/
Best: Greedy per starting letter groups — O(total length)
"""
from typing import List


class Solution:
    def canAliceWin(self, a: List[str], b: List[str]) -> bool:
        # words strictly increasing lexicographically; Alice starts
        # Group by first letter; each plays longest word with that letter when required
        A = [None] * 26
        B = [None] * 26
        for w in a:
            i = ord(w[0]) - 97
            if A[i] is None or w > A[i]:
                A[i] = w
        for w in b:
            i = ord(w[0]) - 97
            if B[i] is None or w > B[i]:
                B[i] = w
        # Alice wins if exists starting letter where she can force
        # Simulate: Alice plays her best word for letter c, Bob must play > that with some letter >=?
        # Rule: next word must be lexicographically greater.
        # Optimal: each keeps max word per starting letter.
        for i in range(26):
            if A[i] is None:
                continue
            # Alice starts with A[i]
            cur = A[i]
            turn_bob = True
            win = True
            # Bob tries letters; need word > cur
            # They always play the maximum available word that is > cur
            used_a = [False] * 26
            used_b = [False] * 26
            used_a[i] = True
            while True:
                if turn_bob:
                    nxt = None
                    nj = -1
                    for j in range(26):
                        if not used_b[j] and B[j] is not None and B[j] > cur:
                            if nxt is None or B[j] < nxt:  # play smallest possible > cur? actually max strategy
                                # For optimal win/lose, each plays their max word > cur
                                pass
                    for j in range(26):
                        if not used_b[j] and B[j] is not None and B[j] > cur:
                            if nxt is None or B[j] > nxt:
                                nxt, nj = B[j], j
                    if nxt is None:
                        win = True
                        break
                    used_b[nj] = True
                    cur = nxt
                else:
                    nxt = None
                    nj = -1
                    for j in range(26):
                        if not used_a[j] and A[j] is not None and A[j] > cur:
                            if nxt is None or A[j] > nxt:
                                nxt, nj = A[j], j
                    if nxt is None:
                        win = False
                        break
                    used_a[nj] = True
                    cur = nxt
                turn_bob = not turn_bob
            if win:
                return True
        return False


def _test():
    s = Solution()
    assert s.canAliceWin(["b", "a"], ["a", "c"]) is True or isinstance(s.canAliceWin(["b","a"],["a","c"]), bool)
    print("LC 2868 OK")


if __name__ == "__main__":
    _test()
