"""
LC 2792. Count Nodes That Are Great Enough — Hard
Best: DFS collect k-th in subtree — O(n log n) with merge
"""

from typing import Optional, List
import heapq


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def countGreatEnoughNodes(self, root: Optional[TreeNode], k: int) -> int:
        self.ans = 0

        def dfs(node):
            if not node:
                return []
            # merge smallest k from children
            a = dfs(node.left)
            b = dfs(node.right)
            heap = a + b
            heapq.heapify(heap)
            # k smallest in subtree excluding self for comparison
            if len(heap) >= k:
                # k-th smallest is max of k smallest
                kth = heapq.nsmallest(k, heap)[-1]
                if node.val > kth:
                    self.ans += 1
            heapq.heappush(heap, node.val)
            return heapq.nsmallest(k, heap)

        dfs(root)
        return self.ans


def _test():
    #   7
    #  / \
    # 6   5  k=2 -> 7 is great (6,5 both <7) => 1
    root = TreeNode(7, TreeNode(6), TreeNode(5))
    assert Solution().countGreatEnoughNodes(root, 2) == 1
    print("LC 2792 OK")


if __name__ == "__main__":
    _test()
