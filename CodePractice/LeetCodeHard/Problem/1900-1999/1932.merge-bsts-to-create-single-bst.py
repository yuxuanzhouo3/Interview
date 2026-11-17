"""
LC 1932. Merge BSTs to Create Single BST — Hard
https://leetcode.com/problems/merge-bsts-to-create-single-bst/
Best: Unique leaf-root merge + inorder check — O(n)
"""

from typing import List, Optional, Dict

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def canMerge(self, trees: List[TreeNode]) -> Optional[TreeNode]:
        roots: Dict[int, TreeNode] = {t.val: t for t in trees}
        seen = set()
        for t in trees:
            if t.left:
                seen.add(t.left.val)
            if t.right:
                seen.add(t.right.val)

        root = None
        for t in trees:
            if t.val not in seen:
                if root is not None:
                    return None
                root = t
        if root is None:
            return None

        def attach(node: TreeNode) -> int:
            """Return leaf count contribution / validate merge count."""
            nonlocal merge_cnt
            if not node:
                return 0
            if not node.left and not node.right and node.val in roots and roots[node.val] is not node:
                child = roots[node.val]
                node.left = child.left
                node.right = child.right
                del roots[node.val]
                merge_cnt += 1
            return 1 + (1 if node.left else 0) + (1 if node.right else 0)

        # iterative stack attach all leaves that match roots
        merge_cnt = 0
        stack = [root]
        visited = {id(root)}
        while stack:
            node = stack.pop()
            for child_attr in ('left', 'right'):
                child = getattr(node, child_attr)
                if child and child.val in roots and roots[child.val] is not child:
                    nxt = roots.pop(child.val)
                    setattr(node, child_attr, nxt)
                    merge_cnt += 1
                    child = nxt
                if child and id(child) not in visited:
                    visited.add(id(child))
                    stack.append(child)

        if merge_cnt != len(trees) - 1:
            return None

        # validate BST via inorder
        prev = -1
        st = []
        cur = root
        cnt = 0
        while st or cur:
            while cur:
                st.append(cur)
                cur = cur.left
            cur = st.pop()
            if cur.val <= prev:
                return None
            prev = cur.val
            cnt += 1
            cur = cur.right
        # all unique values from trees must appear
        # each tree has 1-3 nodes; after merge node count = unique vals
        return root

def _test():

    def build(vals):
        # vals like [2,1] meaning root=2 left=1; [3,2,5] root=3 left=2 right=5
        if not vals:
            return None
        r = TreeNode(vals[0])
        if len(vals) > 1 and vals[1] is not None:
            r.left = TreeNode(vals[1])
        if len(vals) > 2 and vals[2] is not None:
            r.right = TreeNode(vals[2])
        return r

    def to_list(root):
        if not root:
            return []
        from collections import deque
        q = deque([root])
        out = []
        while q:
            n = q.popleft()
            if n is None:
                out.append(None)
                continue
            out.append(n.val)
            if n.left or n.right or True:
                q.append(n.left)
                q.append(n.right)
        while out and out[-1] is None:
            out.pop()
        return out

    s = Solution()
    t1 = [build([2, 1]), build([3, 2, 5]), build([5, 4])]
    r = s.canMerge(t1)
    assert to_list(r) == [3, 2, 5, 1, None, 4]
    t2 = [build([5, 3, 8]), build([3, 2, 6])]
    assert s.canMerge(t2) is None
    t3 = [build([5, 4]), build([3])]
    assert s.canMerge(t3) is None
    print("LC 1932 OK")

if __name__ == "__main__":
    _test()
