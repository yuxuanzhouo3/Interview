"""
LC 3526. Range XOR Queries With Subarray Reversals — Hard
Best: Treap / splay with reverse lazy + xor — O(q log n)
"""

from typing import List
import random

class Node:
    __slots__=["val","xor","prio","left","right","size","rev"]
    def __init__(self, val):
        self.val=val; self.xor=val; self.prio=random.randrange(1<<30)
        self.left=self.right=None; self.size=1; self.rev=False

def sz(t): return t.size if t else 0
def xorv(t): return t.xor if t else 0

def pull(t):
    if not t: return
    t.size=1+sz(t.left)+sz(t.right)
    t.xor=t.val^xorv(t.left)^xorv(t.right)

def push(t):
    if t and t.rev:
        t.left,t.right=t.right,t.left
        if t.left: t.left.rev^=True
        if t.right: t.right.rev^=True
        t.rev=False

def split(t,k):
    if not t: return None,None
    push(t)
    if sz(t.left)>=k:
        a,t.left=split(t.left,k); pull(t); return a,t
    else:
        t.right,b=split(t.right,k-sz(t.left)-1); pull(t); return t,b

def merge(a,b):
    if not a or not b: return a or b
    if a.prio<b.prio:
        push(a); a.right=merge(a.right,b); pull(a); return a
    else:
        push(b); b.left=merge(a,b.left); pull(b); return b

class Solution:
    def reverseAndRangeXorQueries(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        root=None
        for x in nums:
            root=merge(root, Node(x))
        ans=[]
        for q in queries:
            if q[0]==1:
                # reverse [l,r] 1-index
                l,r=q[1],q[2]
                a,b=split(root,l-1)
                b,c=split(b,r-l+1)
                if b: b.rev^=True
                root=merge(a,merge(b,c))
            else:
                l,r=q[1],q[2]
                a,b=split(root,l-1)
                b,c=split(b,r-l+1)
                ans.append(xorv(b))
                root=merge(a,merge(b,c))
        return ans

def _test():
    s=Solution()
    r=s.reverseAndRangeXorQueries([1,2,3], [[2,1,3]])
    assert r==[0]
    print("LC 3526 OK")

if __name__ == "__main__":
    _test()
