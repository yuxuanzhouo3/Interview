"""
LC 272. Closest Binary Search Tree Value II — Hard
https://leetcode.com/problems/closest-binary-search-tree-value-ii/
Best: 双栈前驱/后继，各走 k 步 — O(k + h)
"""

from typing import List, Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def closestKValues(self, root: Optional[TreeNode], target: float, k: int) -> List[int]:
        pred, succ = [], []

        def init_pred(node):
            while node:
                if node.val <= target:
                    pred.append(node)
                    node = node.right
                else:
                    node = node.left

        def init_succ(node):
            while node:
                if node.val > target:
                    succ.append(node)
                    node = node.left
                else:
                    node = node.right

        def next_pred():
            node = pred.pop()
            v = node.val
            node = node.left
            while node:
                pred.append(node)
                node = node.right
            return v

        def next_succ():
            node = succ.pop()
            v = node.val
            node = node.right
            while node:
                succ.append(node)
                node = node.left
            return v

        init_pred(root)
        init_succ(root)
        ans = []
        while k:
            if not pred:
                ans.append(next_succ())
            elif not succ:
                ans.append(next_pred())
            elif abs(pred[-1].val - target) <= abs(succ[-1].val - target):
                ans.append(next_pred())
            else:
                ans.append(next_succ())
            k -= 1
        return ans


def _build(vals):
    """按层序（含 None）建树。"""
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
    # [4,2,5,1,3], target=3.714286, k=2 → [4,3]
    root = _build([4, 2, 5, 1, 3])
    r = sorted(s.closestKValues(root, 3.714286, 2))
    assert r == [3, 4]
    root = _build([1])
    assert s.closestKValues(root, 0.0, 1) == [1]
    print("LC 272 OK")


if __name__ == "__main__":
    _test()
