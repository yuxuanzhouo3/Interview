"""
LC 1948. Delete Duplicate Folders in System — Hard
https://leetcode.com/problems/delete-duplicate-folders-in-system/
Best: Trie serialize subtree mark duplicates — O(N log N)
"""

from collections import defaultdict
from typing import List, Dict

class Trie:
    def __init__(self):
        self.children: Dict[str, "Trie"] = defaultdict(Trie)
        self.deleted: bool = False

class Solution:
    def deleteDuplicateFolder(self, paths: List[List[str]]) -> List[List[str]]:
        root = Trie()
        for path in paths:
            cur = root
            for name in path:
                cur = cur.children[name]

        g: Dict[str, Trie] = {}

        def dfs(node: Trie) -> str:
            if not node.children:
                return ""
            subs = [f"{name}({dfs(child)})" for name, child in node.children.items()]
            s = "".join(sorted(subs))
            if s in g:
                node.deleted = g[s].deleted = True
            else:
                g[s] = node
            return s

        def dfs2(node: Trie) -> None:
            if node.deleted:
                return
            if path:
                ans.append(path[:])
            for name, child in node.children.items():
                path.append(name)
                dfs2(child)
                path.pop()

        dfs(root)
        ans: List[List[str]] = []
        path: List[str] = []
        dfs2(root)
        return ans

def _test():

    s = Solution()
    r1 = s.deleteDuplicateFolder([["a"],["c"],["d"],["a","b"],["c","b"],["d","a"]])
    assert sorted(r1) == sorted([["d"],["d","a"]])
    r2 = s.deleteDuplicateFolder([["a"],["c"],["a","b"],["c","b"],["a","b","x"],["a","b","x","y"],["w"],["w","y"]])
    assert sorted(r2) == sorted([["c"],["c","b"],["a"],["a","b"]])
    r3 = s.deleteDuplicateFolder([["a","b"],["c","d"],["c"],["a"]])
    assert sorted(r3) == sorted([["c"],["c","d"],["a"],["a","b"]])
    print("LC 1948 OK")

if __name__ == "__main__":
    _test()
