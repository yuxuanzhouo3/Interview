"""
LC 124. Binary Tree Maximum Path Sum — Hard
Best: Postorder DFS gain — O(n)
"""

from typing import Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        self.best = float("-inf")

        def gain(node):
            if not node:
                return 0
            left = max(gain(node.left), 0)
            right = max(gain(node.right), 0)
            self.best = max(self.best, node.val + left + right)
            return node.val + max(left, right)

        gain(root)
        return self.best


def _test():
    s = Solution()
    # [-10,9,20,null,null,15,7]
    r = TreeNode(-10, TreeNode(9), TreeNode(20, TreeNode(15), TreeNode(7)))
    assert s.maxPathSum(r) == 42
    assert s.maxPathSum(TreeNode(1, TreeNode(2), TreeNode(3))) == 6
    print("LC 124 OK")


if __name__ == "__main__":
    _test()
