"""
LC 1373. Maximum Sum BST in Binary Tree — Hard
https://leetcode.com/problems/maximum-sum-bst-in-binary-tree/
Best: Postorder return (isBST,min,max,sum) — O(n)
"""

from typing import Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def maxSumBST(self, root: Optional[TreeNode]) -> int:
        self.ans = 0

        def dfs(node):
            if not node:
                return True, float("inf"), float("-inf"), 0
            ls, lmin, lmax, lsum = dfs(node.left)
            rs, rmin, rmax, rsum = dfs(node.right)
            if ls and rs and lmax < node.val < rmin:
                s = lsum + rsum + node.val
                self.ans = max(self.ans, s)
                return True, min(lmin, node.val), max(rmax, node.val), s
            return False, 0, 0, 0

        dfs(root)
        return self.ans


def _build(vals):
    if not vals:
        return None
    nodes = [None if v is None else TreeNode(v) for v in vals]
    kids = nodes[::-1]
    root = kids.pop()
    for n in nodes:
        if n:
            if kids:
                n.left = kids.pop()
            if kids:
                n.right = kids.pop()
    return root


def _test():
    s = Solution()
    assert s.maxSumBST(_build([1, 4, 3, 2, 4, 2, 5, None, None, None, None, None, None, 4, 6])) == 20
    assert s.maxSumBST(_build([4, 3, None, 1, 2])) == 2
    assert s.maxSumBST(_build([-4, -2, -5])) == 0
    print("LC 1373 OK")


if __name__ == "__main__":
    _test()
