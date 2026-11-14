"""
LC 212. Word Search II — Hard
Best: Trie + DFS prune — O(m*n*4^L)
"""

from typing import List


class TrieNode:
    __slots__ = ("children", "word")
    def __init__(self):
        self.children = {}
        self.word = None


class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        root = TrieNode()
        for w in words:
            node = root
            for c in w:
                node = node.children.setdefault(c, TrieNode())
            node.word = w
        m, n = len(board), len(board[0])
        ans = []

        def dfs(i, j, node):
            c = board[i][j]
            nxt = node.children.get(c)
            if not nxt:
                return
            if nxt.word:
                ans.append(nxt.word)
                nxt.word = None
            board[i][j] = "#"
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n and board[ni][nj] != "#":
                    dfs(ni, nj, nxt)
            board[i][j] = c
            if not nxt.children:
                node.children.pop(c, None)

        for i in range(m):
            for j in range(n):
                dfs(i, j, root)
        return ans


def _test():
    s = Solution()
    board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]]
    r = s.findWords(board, ["oath", "pea", "eat", "rain"])
    assert sorted(r) == ["eat", "oath"]
    print("LC 212 OK")


if __name__ == "__main__":
    _test()
