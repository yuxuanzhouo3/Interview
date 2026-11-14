"""
LC 420. Strong Password Checker — Hard
Best: Case by length + replace for repeats — O(n)
"""

class Solution:
    def strongPasswordChecker(self, password: str) -> int:
        n = len(password)
        missing = 3
        if any(c.islower() for c in password):
            missing -= 1
        if any(c.isupper() for c in password):
            missing -= 1
        if any(c.isdigit() for c in password):
            missing -= 1

        # repeat runs
        replace = 0
        one = two = 0
        i = 2
        while i < n:
            if password[i] == password[i - 1] == password[i - 2]:
                length = 2
                while i < n and password[i] == password[i - 1]:
                    length += 1
                    i += 1
                replace += length // 3
                if length % 3 == 0:
                    one += 1
                elif length % 3 == 1:
                    two += 1
            else:
                i += 1

        if n < 6:
            return max(6 - n, missing)
        if n <= 20:
            return max(replace, missing)

        delete = n - 20
        replace -= min(delete, one)
        replace -= min(max(delete - one, 0), two * 2) // 2
        replace -= max(delete - one - 2 * two, 0) // 3
        return delete + max(replace, missing)


def _test():
    s = Solution()
    assert s.strongPasswordChecker("a") == 5
    assert s.strongPasswordChecker("aA1") == 3
    assert s.strongPasswordChecker("1337C0d3") == 0
    print("LC 420 OK")


if __name__ == "__main__":
    _test()
