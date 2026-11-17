"""
LC 3383. Minimum Runes to Add to Cast Spell — Hard
Best: SCC + indegree of components with crystals — O(n+m)
"""

from typing import List
from collections import defaultdict

class Solution:
    def minRunesToAdd(self, n: int, crystals: List[int], flowFrom: List[int], flowTo: List[int]) -> int:
        g=defaultdict(list); rg=defaultdict(list)
        for a,b in zip(flowFrom, flowTo):
            g[a].append(b); rg[b].append(a)
        # Kosaraju SCC
        seen=[False]*n; order=[]
        def dfs1(u):
            seen[u]=True
            for v in g[u]:
                if not seen[v]: dfs1(v)
            order.append(u)
        for i in range(n):
            if not seen[i]: dfs1(i)
        comp=[-1]*n; cid=0
        def dfs2(u,c):
            comp[u]=c
            for v in rg[u]:
                if comp[v]<0: dfs2(v,c)
        for u in reversed(order):
            if comp[u]<0:
                dfs2(u,cid); cid+=1
        has=[False]*cid
        for c in crystals:
            has[comp[c]]=True
        indeg=[0]*cid
        for a,b in zip(flowFrom, flowTo):
            if comp[a]!=comp[b]:
                indeg[comp[b]]+=1
        # need add edges so all reachable from some crystal; add runes = new nodes with crystal into indeg0 non-crystal comps
        ans=0
        for c in range(cid):
            if indeg[c]==0 and not has[c]:
                ans+=1
        return ans

def _test():
    s=Solution()
    r=s.minRunesToAdd(3,[0],[0,1,2],[1,2,0])
    print("LC 3383", r, "OK")

if __name__ == "__main__":
    _test()
