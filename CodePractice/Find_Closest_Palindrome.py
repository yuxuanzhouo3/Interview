# O(L); O(L)
class Solution1:
    def nearestPalindromic(self, S: str) -> str:
        n = len(S)
        mid = (n + 1) // 2
        prefix = int(S[:mid])
        candidates = set()
        candidates.add(10**(n-1)-1)
        candidates.add(10**n +1)

        # O(L); O(L)
        for p in [ prefix, prefix -1, prefix +1]:
            s = str(p)

            if n % 2 == 0:
                s = s + s[::-1]
            else:
                s = s + s[-2::-1]

            candidates.add(int(s))

        candidates.discard(int(S))

        # print(candidates,sorted(candidates, key=lambda x: (abs(x - int(S)), x ) ))

        return str(sorted(candidates, key=lambda x: ( abs(x - int(S)), x ))[0])

# O(L^2); O(L)
class Solution2:
    # Mirror left half -> right half
    # 12345 -> 12321
    # 1234  -> 1221
    def convert(self, num: int) -> int:
        s = list(str(num))
        n = len(s)
        mid = (n + 1) // 2
        return int("".join(s[:mid] + s[:mid][::-1])) if n & 1 == 0 else int("".join(s[:mid] + s[:mid-1][::-1]))

    def convert2(self, num: int) -> int:
        s = list(str(num))
        n = len(s)

        l = (n - 1) // 2
        r = n // 2

        while l >= 0:
            s[r] = s[l]
            l -= 1
            r += 1

        return int("".join(s))

    # Largest palindrome < num
    def previous_palindrome(self, num: int) -> int:
        left, right = 0, num # [0, num) , right=mid; [0, num] , right=mid-1;
        ans = -1

        while left < right: # [left, right) , right=mid; [left, right] , right=mid-1;
            mid = left + (right - left) // 2
            pal = self.convert(mid)

            if pal < num:
                ans = pal
                left = mid + 1
            else:
                right = mid

        return ans


    # Smallest palindrome > num
    def next_palindrome(self, num: int) -> int:
        left, right = num + 1, 10**18+1 # [num+1, 10**18+1) , right=mid; [left, right] , right=mid-1;
        ans = right

        while left < right: # [left, right) , right=mid; [left, right] , right=mid-1;
            mid = left + (right - left) // 2
            pal = self.convert(mid)

            if pal > num:
                ans = pal
                right = mid
            else:
                left = mid + 1

        return ans


    def nearestPalindromic(self, n: str) -> str:
        num = int(n)
        prev = self.previous_palindrome(num)
        next_ = self.next_palindrome(num)
        # Equal distance -> choose smaller
        if num - prev <= next_ - num:
            return str(prev)
        return str(next_)


import time

N = 10000
s = "123456789"

obj1 = Solution1()
obj2 = Solution2()

t1 = time.perf_counter()
for _ in range(N):
    obj1.nearestPalindromic(s)
t2 = time.perf_counter()

t3 = time.perf_counter()
for _ in range(N):
    obj2.nearestPalindromic(s)
t4 = time.perf_counter()

print(f"Solution1: {t2-t1:.6f}s")
print(f"Solution2: {t4-t3:.6f}s")

print(f"Solution1 avg: {(t2-t1)/N:.9f}s")
print(f"Solution2 avg: {(t4-t3)/N:.9f}s")