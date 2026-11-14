"""
LC 968. Binary Tree Cameras — Hard
https://leetcode.com/problems/binary-tree-cameras/
Best: Greedy DFS states — O(n)
"""

from typing import Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def minCameraCover(self, root: Optional[TreeNode]) -> int:
        # 0: need cover, 1: has camera, 2: covered
        self.ans = 0

        def dfs(node):
            if not node:
                return 2
            L, R = dfs(node.left), dfs(node.right)
            if L == 0 or R == 0:
                self.ans += 1
                return 1
            if L == 1 or R == 1:
                return 2
            return 0

        if dfs(root) == 0:
            self.ans += 1
        return self.ans


def _test():
    s = Solution()
    # [0,0,null,0,0] -> 1
    r = TreeNode(0, TreeNode(0, TreeNode(0), TreeNode(0)))
    assert s.minCameraCover(r) == 1
    # [0,0,null,0,null,0,null,null,0] -> 2
    r2 = TreeNode(0, TreeNode(0, TreeNode(0, TreeNode(0, None, TreeNode(0)))))
    assert s.minCameraCover(r2) == 2
    assert s.minCameraCover(TreeNode(0)) == 1
    print("LC 968 OK")


if __name__ == "__main__":
    _test()
