"""
LC 2227. Encrypt and Decrypt Strings — Hard
https://leetcode.com/problems/encrypt-and-decrypt-strings/
Best: Map encrypt + Counter dictionary — O(Σ|w|)
"""

from collections import Counter
from typing import List

class Encrypter:
    def __init__(self, keys: List[str], values: List[str], dictionary: List[str]):
        self.mp = dict(zip(keys, values))
        self.cnt = Counter(self.encrypt(v) for v in dictionary)

    def encrypt(self, word1: str) -> str:
        res = []
        for c in word1:
            if c not in self.mp:
                return ""
            res.append(self.mp[c])
        return "".join(res)

    def decrypt(self, word2: str) -> int:
        return self.cnt[word2]

def _test():

    e = Encrypter(['a','b','c','d'], ["ei","zf","ei","am"], ["abcd","acbd","adbc","badc","dacb","cadb","cbda","abad"])
    assert e.encrypt("abcd") == "eizfeiam"
    assert e.decrypt("eizfeiam") == 2
    print("LC 2227 OK")

if __name__ == "__main__":
    _test()
