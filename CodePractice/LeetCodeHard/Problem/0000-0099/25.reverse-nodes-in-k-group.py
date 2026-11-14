"""
LC 25. Reverse Nodes in k-Group — Hard
Best: Iterative reverse by groups — O(n)
"""

from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        dummy = ListNode(0, head)
        group_prev = dummy

        while True:
            kth = group_prev
            for _ in range(k):
                kth = kth.next
                if not kth:
                    return dummy.next
            group_next = kth.next

            # reverse [group_prev.next .. kth]
            prev, cur = group_next, group_prev.next
            while cur is not group_next:
                nxt = cur.next
                cur.next = prev
                prev = cur
                cur = nxt

            tmp = group_prev.next
            group_prev.next = kth
            group_prev = tmp


def _build(a):
    d = c = ListNode()
    for x in a:
        c.next = ListNode(x)
        c = c.next
    return d.next


def _tolist(h):
    r = []
    while h:
        r.append(h.val)
        h = h.next
    return r


def _test():
    s = Solution()
    assert _tolist(s.reverseKGroup(_build([1, 2, 3, 4, 5]), 2)) == [2, 1, 4, 3, 5]
    assert _tolist(s.reverseKGroup(_build([1, 2, 3, 4, 5]), 3)) == [3, 2, 1, 4, 5]
    print("LC 25 OK")


if __name__ == "__main__":
    _test()
