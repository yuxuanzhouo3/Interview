"""
LC 1597. Build Binary Expression Tree From Infix Expression — Hard
https://leetcode.com/problems/build-binary-expression-tree-from-infix-expression/
Best: 双栈按优先级建树 — O(n)
"""


class Node:
    def __init__(self, val=" ", left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def expTree(self, s: str) -> "Node":
        prec = {"+": 1, "-": 1, "*": 2, "/": 2}
        nodes, ops = [], []

        def build():
            op = ops.pop()
            r, l = nodes.pop(), nodes.pop()
            nodes.append(Node(op, l, r))

        for c in s:
            if c.isdigit():
                nodes.append(Node(c))
            elif c == "(":
                ops.append(c)
            elif c == ")":
                while ops[-1] != "(":
                    build()
                ops.pop()
            else:
                while ops and ops[-1] != "(" and prec[ops[-1]] >= prec[c]:
                    build()
                ops.append(c)
        while ops:
            build()
        return nodes[-1]


def _inorder(node):
    if not node:
        return ""
    if node.left is None and node.right is None:
        return node.val
    return _inorder(node.left) + node.val + _inorder(node.right)


def _eval(node):
    if node.left is None:
        return int(node.val)
    l, r = _eval(node.left), _eval(node.right)
    if node.val == "+":
        return l + r
    if node.val == "-":
        return l - r
    if node.val == "*":
        return l * r
    return int(l / r)


def _test():
    s = Solution()
    t = s.expTree("3*4-2*5")
    assert _inorder(t) == "3*4-2*5"
    assert _eval(t) == 2
    t2 = s.expTree("2-3/(5*2)+1")
    assert _eval(t2) == 3
    t3 = s.expTree("(2-3)*(5+2)+1")
    assert _eval(t3) == -6
    print("LC 1597 OK")


if __name__ == "__main__":
    _test()
