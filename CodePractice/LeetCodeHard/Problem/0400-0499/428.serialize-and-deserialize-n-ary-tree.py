"""
LC 428. Serialize and Deserialize N-ary Tree — Hard
https://leetcode.com/problems/serialize-and-deserialize-n-ary-tree/
Best: 前序 + 子节点个数编码 — O(n)
"""

from typing import List, Optional


class Node:
    def __init__(self, val: int = None, children: List["Node"] = None):
        self.val = val
        self.children = children if children is not None else []


class Codec:
    def serialize(self, root: "Node") -> str:
        if not root:
            return ""
        parts = []

        def dfs(node: Node):
            parts.append(str(node.val))
            parts.append(str(len(node.children)))
            for ch in node.children:
                dfs(ch)

        dfs(root)
        return ",".join(parts)

    def deserialize(self, data: str) -> "Node":
        if not data:
            return None
        vals = data.split(",")
        self.i = 0

        def dfs() -> Node:
            val = int(vals[self.i])
            self.i += 1
            nchild = int(vals[self.i])
            self.i += 1
            node = Node(val, [])
            for _ in range(nchild):
                node.children.append(dfs())
            return node

        return dfs()


def _same(a: Optional[Node], b: Optional[Node]) -> bool:
    if a is None or b is None:
        return a is b
    if a.val != b.val or len(a.children) != len(b.children):
        return False
    return all(_same(x, y) for x, y in zip(a.children, b.children))


def _test():
    codec = Codec()
    # 1 → 3,2,4 ; 3 → 5,6
    root = Node(1, [Node(3, [Node(5), Node(6)]), Node(2), Node(4)])
    data = codec.serialize(root)
    assert _same(codec.deserialize(data), root)
    assert codec.serialize(None) == ""
    assert codec.deserialize("") is None
    single = Node(1)
    assert _same(codec.deserialize(codec.serialize(single)), single)
    print("LC 428 OK")


if __name__ == "__main__":
    _test()
