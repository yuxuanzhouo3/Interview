"""
LC 126. Word Ladder II — Hard
Best: BFS build parents + DFS collect paths
"""

from typing import List
from collections import defaultdict, deque


class Solution:
    def findLadders(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
        words = set(wordList)
        if endWord not in words:
            return []
        words.add(beginWord)
        parents = defaultdict(set)
        q = deque([beginWord])
        found = False
        visited_level = set()
        while q and not found:
            level_vis = set()
            for _ in range(len(q)):
                w = q.popleft()
                for i in range(len(w)):
                    for c in "abcdefghijklmnopqrstuvwxyz":
                        nw = w[:i] + c + w[i + 1:]
                        if nw not in words:
                            continue
                        if nw not in visited_level:
                            if nw not in level_vis:
                                level_vis.add(nw)
                                q.append(nw)
                            parents[nw].add(w)
                            if nw == endWord:
                                found = True
            visited_level |= level_vis
            words -= level_vis
        if not found:
            return []
        ans = []
        path = [endWord]

        def dfs(w):
            if w == beginWord:
                ans.append(path[::-1])
                return
            for p in parents[w]:
                path.append(p)
                dfs(p)
                path.pop()

        dfs(endWord)
        return ans


def _test():
    s = Solution()
    r = s.findLadders("hit", "cog", ["hot", "dot", "dog", "lot", "log", "cog"])
    assert sorted(r) == sorted([["hit","hot","dot","dog","cog"],["hit","hot","lot","log","cog"]])
    print("LC 126 OK")


if __name__ == "__main__":
    _test()
