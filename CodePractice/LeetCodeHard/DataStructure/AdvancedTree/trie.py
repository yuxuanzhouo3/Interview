"""
10. Trie (Prefix Tree)
Template: query / update / add / del
App: 自动补全、词频、IP 路由最长前缀、敏感词；Redis CLUSTER slot 思想无关，但搜索场景极常用。
"""


class TrieNode:
    def __init__(self):
        self.children = {}
        self.val = None  # end marker payload
        self.cnt = 0     # words ending here


class Trie:
    def __init__(self):
        self.root = TrieNode()

    # ---------- add ----------
    def add(self, word, val=True):
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.cnt += 1
        node.val = val

    # ---------- query ----------
    def query(self, word):
        node = self._walk(word)
        return None if not node or node.cnt == 0 else node.val

    def starts_with(self, prefix):
        return self._walk(prefix) is not None

    def _walk(self, s):
        node = self.root
        for ch in s:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node

    # ---------- update ----------
    def update(self, word, val):
        node = self._walk(word)
        if not node or node.cnt == 0:
            return False
        node.val = val
        return True

    # ---------- del ----------
    def delete(self, word):
        def dfs(node, i):
            if i == len(word):
                if node.cnt == 0:
                    return False
                node.cnt -= 1
                node.val = None if node.cnt == 0 else node.val
                return node.cnt == 0 and not node.children
            ch = word[i]
            if ch not in node.children:
                return False
            can_remove = dfs(node.children[ch], i + 1)
            if can_remove:
                del node.children[ch]
            return node.cnt == 0 and not node.children and node is not self.root

        dfs(self.root, 0)


def _test():
    t = Trie()
    for w in ["app", "apple", "ape", "bat"]:
        t.add(w, w.upper())
    assert t.query("app") == "APP"
    assert t.starts_with("ap")
    assert t.update("ape", "APE2") and t.query("ape") == "APE2"
    t.delete("app")
    assert t.query("app") is None
    assert t.query("apple") == "APPLE"
    print("Trie OK")


APPLICATION = """
- 搜索建议、拼写检查、九宫格输入；LeetCode 208/212。
- 网络: IP 最长前缀匹配（可压缩成 Patricia）。
- vs HashMap: Trie 共享前缀、有序枚举前缀；哈希只能精确键。
"""

if __name__ == "__main__":
    _test()
    print(APPLICATION)
