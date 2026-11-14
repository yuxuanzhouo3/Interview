"""
LC 297. Serialize and Deserialize Binary Tree — Hard
Best: Preorder DFS with null markers — O(n)
"""

from typing import Optional
from collections import deque


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Codec:
    def serialize(self, root: Optional[TreeNode]) -> str:
        out = []

        def dfs(node):
            if not node:
                out.append("#")
                return
            out.append(str(node.val))
            dfs(node.left)
            dfs(node.right)

        dfs(root)
        return ",".join(out)

    def deserialize(self, data: str) -> Optional[TreeNode]:
        vals = deque(data.split(","))

        def dfs():
            v = vals.popleft()
            if v == "#":
                return None
            node = TreeNode(int(v))
            node.left = dfs()
            node.right = dfs()
            return node

        return dfs()


def _test():
    c = Codec()
    root = TreeNode(1, TreeNode(2), TreeNode(3, TreeNode(4), TreeNode(5)))
    assert c.serialize(c.deserialize(c.serialize(root))) == c.serialize(root)
    print("LC 297 OK")


if __name__ == "__main__":
    _test()
