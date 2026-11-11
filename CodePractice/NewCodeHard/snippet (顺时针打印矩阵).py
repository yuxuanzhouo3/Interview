# -*- coding:utf-8 -*-

class Solution:
    def clockwisePrint(self, A, n, m):
        # write code here
        right = m - 1
        down = n - 1
        left = 0
        up = 0
        res = []
        i = 0
        j = 0

        while left < right and up < down: # break ealier to avoid pointers mess up
            j = left
            # left -> right
            while j < right:
                res.append(A[up][j])
                j += 1

            i = up
            # up -> down
            while i < down:
                res.append(A[i][right])
                i += 1

            # down -> left
            while j > left:
                res.append(A[down][j])
                j -= 1

            # left -> up
            while i > up:
                res.append(A[i][left])
                i -= 1

            right -= 1
            down -= 1
            left += 1
            up += 1

        
        # one row remains
        if up == down:
            while left <= right:
                res.append(A[up][left])
                left += 1

        # one column remains
        elif left == right:
            while up <= down:
                res.append(A[up][left])
                up += 1

        return res