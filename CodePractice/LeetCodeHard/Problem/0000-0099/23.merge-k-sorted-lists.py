"""
LC 23. Merge k Sorted Lists — Hard
Best: Min-heap — O(N log k)
"""

from typing import List, Optional
import heapq


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        heap = []
        for i, node in enumerate(lists):
            if node:
                heapq.heappush(heap, (node.val, i, node))
        dummy = cur = ListNode()
        while heap:
            val, i, node = heapq.heappop(heap)
            cur.next = node
            cur = cur.next
            if node.next:
                heapq.heappush(heap, (node.next.val, i, node.next))
        return dummy.next


def _build(arrs):
    out = []
    for a in arrs:
        dummy = cur = ListNode()
        for x in a:
            cur.next = ListNode(x)
            cur = cur.next
        out.append(dummy.next)
    return out


def _tolist(h):
    r = []
    while h:
        r.append(h.val)
        h = h.next
    return r


def _test():
    s = Solution()
    assert _tolist(s.mergeKLists(_build([[1, 4, 5], [1, 3, 4], [2, 6]]))) == [1, 1, 2, 3, 4, 4, 5, 6]
    assert _tolist(s.mergeKLists([])) == []
    print("LC 23 OK")


if __name__ == "__main__":
    _test()
