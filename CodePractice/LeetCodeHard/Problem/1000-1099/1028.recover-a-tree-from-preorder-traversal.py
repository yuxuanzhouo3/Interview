"""
LC 1028. Recover a Tree From Preorder Traversal — Hard
https://leetcode.com/problems/recover-a-tree-from-preorder-traversal/
Best: Stack by depth — O(n)
"""

from typing import Optional, List


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def recoverFromPreorder(self, traversal: str) -> Optional[TreeNode]:
        stack: List[TreeNode] = []
        i, n = 0, len(traversal)
        while i < n:
            depth = 0
            while i < n and traversal[i] == "-":
                depth += 1
                i += 1
            val = 0
            while i < n and traversal[i].isdigit():
                val = val * 10 + int(traversal[i])
                i += 1
            node = TreeNode(val)
            while len(stack) > depth:
                stack.pop()
            if stack:
                if stack[-1].left is None:
                    stack[-1].left = node
                else:
                    stack[-1].right = node
            stack.append(node)
        return stack[0] if stack else None


def _serialize(root):
    if not root:
        return []
    out, q = [], [root]
    while q:
        node = q.pop(0)
        if node:
            out.append(node.val)
            q.append(node.left)
            q.append(node.right)
        else:
            out.append(None)
    while out and out[-1] is None:
        out.pop()
    return out


def _test():
    s = Solution()
    r = s.recoverFromPreorder("1-2--3--4-5--6--7")
    assert _serialize(r) == [1, 2, 5, 3, 4, 6, 7]
    r2 = s.recoverFromPreorder("1-2--3---4-5--6---7")
    assert _serialize(r2) == [1, 2, 5, 3, None, 6, None, 4, None, 7]
    r3 = s.recoverFromPreorder("1-401--349---90--88")
    assert _serialize(r3) == [1, 401, None, 349, 88, 90]
    print("LC 1028 OK")


if __name__ == "__main__":
    _test()
