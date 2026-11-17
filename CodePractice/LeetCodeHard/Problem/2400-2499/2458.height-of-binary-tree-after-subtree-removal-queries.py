"""
LC 2458. Height of Binary Tree After Subtree Removal Queries — Hard
https://leetcode.com/problems/height-of-binary-tree-after-subtree-removal-queries/
Best: Two DFS / precompute left-right max heights — O(n+q)
"""

from typing import List, Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def treeQueries(self, root: Optional[TreeNode], queries: List[int]) -> List[int]:
        height = {}

        def get_h(node):
            if not node:
                return -1
            h = 1 + max(get_h(node.left), get_h(node.right))
            height[node.val] = h
            return h

        get_h(root)
        ans_map = {}

        def dfs(node, depth, rest):
            if not node:
                return
            ans_map[node.val] = rest
            lh = height[node.left.val] if node.left else -1
            rh = height[node.right.val] if node.right else -1
            if node.left:
                dfs(node.left, depth + 1, max(rest, depth + 1 + rh))
            if node.right:
                dfs(node.right, depth + 1, max(rest, depth + 1 + lh))

        dfs(root, 0, 0)
        return [ans_map[q] for q in queries]


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
    root = _build([1, 3, 4, 2, None, 6, 5, None, None, None, None, None, 7])
    assert s.treeQueries(root, [4]) == [2]
    root2 = _build([5, 8, 9, 2, 1, 3, 7, 4, 6])
    assert s.treeQueries(root2, [3, 2, 4, 8]) == [3, 2, 3, 2]
    print("LC 2458 OK")


if __name__ == "__main__":
    _test()
