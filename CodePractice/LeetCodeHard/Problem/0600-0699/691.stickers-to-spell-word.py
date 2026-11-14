"""
LC 691. Stickers to Spell Word — Hard
https://leetcode.com/problems/stickers-to-spell-word/
Best: BFS / DP on target residual mask — O(2^n * S)
"""

from collections import Counter, deque
from typing import List


class Solution:
    def minStickers(self, stickers: List[str], target: str) -> int:
        n = len(target)
        # map target char index for residual count as tuple
        tcnt = Counter(target)
        keys = list(tcnt.keys())
        # filter stickers to useful chars
        sticks = []
        for st in stickers:
            c = Counter(ch for ch in st if ch in tcnt)
            if c:
                sticks.append(c)

        start = tuple(tcnt[k] for k in keys)
        goal = tuple(0 for _ in keys)
        q = deque([(start, 0)])
        seen = {start}

        while q:
            state, dist = q.popleft()
            if state == goal:
                return dist
            for st in sticks:
                # must reduce first positive needed that sticker has? prune: sticker must cover state[0]>0 if has keys[0] etc
                if not any(st[keys[i]] and state[i] for i in range(len(keys))):
                    continue
                nxt = list(state)
                for i, k in enumerate(keys):
                    nxt[i] = max(0, state[i] - st[k])
                t = tuple(nxt)
                if t not in seen:
                    seen.add(t)
                    q.append((t, dist + 1))
        return -1


def _test():
    s = Solution()
    assert s.minStickers(["with", "example", "science"], "thehat") == 3
    assert s.minStickers(["notice", "possible"], "basicbasic") == -1
    print("LC 691 OK")


if __name__ == "__main__":
    _test()
