"""
LC 2534. Time Taken to Cross the Door — Hard
https://leetcode.com/problems/time-taken-to-cross-the-door/
Best: Two queues simulation — O(n)
"""
from typing import List
from collections import deque


class Solution:
    def timeTaken(self, arrival: List[int], state: List[int]) -> List[int]:
        n = len(arrival)
        enter, exit_ = deque(), deque()
        for i in range(n):
            (exit_ if state[i] == 1 else enter).append(i)
        ans = [0] * n
        time = 0
        prev = 1  # last used for exit preferred initially
        i_arr = 0
        waiting_enter, waiting_exit = deque(), deque()
        idxs = list(range(n))
        # better: process by time
        enter_q, exit_q = deque(), deque()
        i = 0
        time = 0
        last = 1  # 1 means exit was last (or door unused -> prefer exit)
        while i < n or enter_q or exit_q:
            while i < n and arrival[i] <= time:
                if state[i] == 0:
                    enter_q.append(i)
                else:
                    exit_q.append(i)
                i += 1
            if not enter_q and not exit_q:
                if i < n:
                    time = arrival[i]
                    last = 1
                continue
            if last == 1:
                if exit_q:
                    p = exit_q.popleft()
                    ans[p] = time
                    last = 1
                else:
                    p = enter_q.popleft()
                    ans[p] = time
                    last = 0
            else:
                if enter_q:
                    p = enter_q.popleft()
                    ans[p] = time
                    last = 0
                else:
                    p = exit_q.popleft()
                    ans[p] = time
                    last = 1
            time += 1
        return ans


def _test():
    s = Solution()
    assert s.timeTaken([0, 1, 1, 2, 4], [0, 1, 0, 0, 1]) == [0, 3, 1, 2, 4]
    assert s.timeTaken([0, 0, 0], [1, 0, 1]) == [0, 2, 1]
    print("LC 2534 OK")


if __name__ == "__main__":
    _test()
