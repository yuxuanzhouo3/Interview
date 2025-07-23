import collections
class TrieNode(object):
    def __init__(self, val):
        self.val = val
        # self.children = collections.defaultdict(lambda: TrieNode(""))
        # self.children = {}
        self.children = collections.defaultdict(TrieNode)
        self.word_prix = []

class Trie(object):
    def __init__(self, Strs):
        self.root = TrieNode("")
        self.build(Strs)
        self.words = Strs

    def build(self, Strs):
        for i, Str in enumerate(Strs):
            node = self.root
            for ch in Str:
                if ch not in node.children:
                    node.children[ch] = TrieNode(ch)
                node = node.children[ch]
                node.word_prix.append(i)
            # node.isWord = True
    def search(self, query):
        node = self.root
        for ch in query:
            if ch not in node.children:
                return []
            node = node.children[ch]
        return [(i, self.words[i]) for i in node.word_prix]


Strs = ["too", "tio", "topl", "sere", "serp", "xyz", "tz", "tx"]
trie = Trie(Strs)
for usr in ["t", "ti", "to", "xyz", "ser"]:
    print(trie.search(usr))