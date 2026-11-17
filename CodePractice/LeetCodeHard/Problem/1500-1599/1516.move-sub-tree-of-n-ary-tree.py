"""
LC 1516. Move Sub-Tree of N-Ary Tree — Hard
Best: Find nodes; handle ancestor cases; rewire — O(n)
"""

class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children if children is not None else []


class Solution:
    def moveSubTree(self, root: "Node", p: "Node", q: "Node") -> "Node":
        if p in q.children:
            return root
        parent = {root: None}
        stack = [root]
        while stack:
            u = stack.pop()
            for v in u.children:
                parent[v] = u
                stack.append(v)

        def is_ancestor(a, b):
            while b:
                if b is a:
                    return True
                b = parent.get(b)
            return False

        # detach p
        pp = parent[p]
        if pp:
            pp.children.remove(p)
        if is_ancestor(p, q):
            # q is under p: move q to p's old place
            pq = parent[q]
            if pq:
                pq.children.remove(q)
            if pp:
                pp.children.append(q)
            parent[q] = pp
            q.children.append(p)
            parent[p] = q
            if p is root:
                root = q
        else:
            q.children.append(p)
            parent[p] = q
        return root


def _test():
    # p already child of q
    q = Node(2); p = Node(1); q.children = [p]; root = q
    r = Solution().moveSubTree(root, p, q)
    assert r is q and p in q.children
    print("LC 1516 OK")


if __name__ == "__main__":
    _test()
