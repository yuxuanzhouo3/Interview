"""
LC 987. Vertical Order Traversal of a Binary Tree — Hard
https://leetcode.com/problems/vertical-order-traversal-of-a-binary-tree/
Best: DFS collect + sort — O(n log n)
"""

from typing import List, Optional
from collections import defaultdict


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def verticalTraversal(self, root: Optional[TreeNode]) -> List[List[int]]:
        cols = defaultdict(list)

        def dfs(node, r, c):
            if not node:
                return
            cols[c].append((r, node.val))
            dfs(node.left, r + 1, c - 1)
            dfs(node.right, r + 1, c + 1)

        dfs(root, 0, 0)
        ans = []
        for c in sorted(cols):
            ans.append([v for _, v in sorted(cols[c])])
        return ans


def _test():
    s = Solution()
    # [3,9,20,null,null,15,7]
    root = TreeNode(3, TreeNode(9), TreeNode(20, TreeNode(15), TreeNode(7)))
    assert s.verticalTraversal(root) == [[9], [3, 15], [20], [7]]
    root2 = TreeNode(1, TreeNode(2, TreeNode(4), TreeNode(5)), TreeNode(3, TreeNode(6), TreeNode(7)))
    assert s.verticalTraversal(root2) == [[4], [2], [1, 5, 6], [3], [7]]
    print("LC 987 OK")


if __name__ == "__main__":
    _test()
