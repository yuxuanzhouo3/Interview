"""
LC 431. Encode N-ary Tree to Binary Tree — Hard
https://leetcode.com/problems/encode-n-ary-tree-to-binary-tree/
Best: 左孩子-右兄弟编码 — O(n)
"""

from typing import List, Optional


class Node:
    def __init__(self, val: int = None, children: List["Node"] = None):
        self.val = val
        self.children = children if children is not None else []


class TreeNode:
    def __init__(self, x: int = 0, left: "TreeNode" = None, right: "TreeNode" = None):
        self.val = x
        self.left = left
        self.right = right


class Codec:
    def encode(self, root: "Node") -> Optional[TreeNode]:
        if not root:
            return None
        bt = TreeNode(root.val)
        if not root.children:
            return bt
        bt.left = self.encode(root.children[0])
        cur = bt.left
        for ch in root.children[1:]:
            cur.right = self.encode(ch)
            cur = cur.right
        return bt

    def decode(self, data: Optional[TreeNode]) -> "Node":
        if not data:
            return None
        root = Node(data.val, [])
        cur = data.left
        while cur:
            root.children.append(self.decode(cur))
            cur = cur.right
        return root


def _same_n(a: Optional[Node], b: Optional[Node]) -> bool:
    if a is None or b is None:
        return a is b
    if a.val != b.val or len(a.children) != len(b.children):
        return False
    return all(_same_n(x, y) for x, y in zip(a.children, b.children))


def _test():
    codec = Codec()
    root = Node(1, [Node(3, [Node(5), Node(6)]), Node(2), Node(4)])
    enc = codec.encode(root)
    assert enc.val == 1 and enc.left.val == 3
    assert _same_n(codec.decode(enc), root)
    assert codec.encode(None) is None
    assert codec.decode(None) is None
    print("LC 431 OK")


if __name__ == "__main__":
    _test()
