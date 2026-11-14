"""
LC 1032. Stream of Characters — Hard
https://leetcode.com/problems/stream-of-characters/
Best: Reverse Trie — O(QL)
"""

from typing import List


class StreamChecker:
    def __init__(self, words: List[str]):
        self.trie = {}
        self.stream = []
        self.max_len = 0
        for w in words:
            node = self.trie
            for ch in reversed(w):
                node = node.setdefault(ch, {})
            node["#"] = True
            self.max_len = max(self.max_len, len(w))

    def query(self, letter: str) -> bool:
        self.stream.append(letter)
        node = self.trie
        for i in range(len(self.stream) - 1, max(-1, len(self.stream) - self.max_len - 1), -1):
            ch = self.stream[i]
            if ch not in node:
                return False
            node = node[ch]
            if "#" in node:
                return True
        return False


def _test():
    sc = StreamChecker(["cd", "f", "kl"])
    assert sc.query("a") is False
    assert sc.query("b") is False
    assert sc.query("c") is False
    assert sc.query("d") is True
    assert sc.query("e") is False
    assert sc.query("f") is True
    assert sc.query("g") is False
    assert sc.query("h") is False
    assert sc.query("i") is False
    assert sc.query("j") is False
    assert sc.query("k") is False
    assert sc.query("l") is True
    print("LC 1032 OK")


if __name__ == "__main__":
    _test()
