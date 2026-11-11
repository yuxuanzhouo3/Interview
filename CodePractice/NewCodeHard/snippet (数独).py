import sys

class Solution:
    def __init__(self, A):
        self.A = A

    def solveSudoku(self) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        A = self.A
        if not A: return 
        self.N = len(A)
        self.M = len(A[0])
        self.dfs(A)
    
    def dfs(self, A: list[list[str]]) -> bool:
        for i in range(self.N):
            for j in range(self.M):
                if A[i][j] != '0': continue
                for v in "123456789":
                    if not self.isVaild(A, i, j, v): continue
                    A[i][j] = v
                    if self.dfs(A): return True
                    A[i][j] = '0'
                return False
        return True


    def isVaild(self, A: list[list[str]], i: int, j: int, v: str) -> bool:
        for c in range(self.M):
            if A[i][c] == v: return False
        for r in range(self.N):
            if A[r][j] == v: return False
        # left-top 
        row = i - i % 3; col = j - j % 3
        for k in range(9): 
            if A[row+k//3][col+k%3] == v: return False
        return True

input = sys.stdin.readline
A = [list(input().split()) for _ in range(9)]

solv = Solution(A)      
solv.solveSudoku()

for r in solv.A:
    print(*r)