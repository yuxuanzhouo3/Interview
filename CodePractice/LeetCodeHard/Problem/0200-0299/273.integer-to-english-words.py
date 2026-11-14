"""
LC 273. Integer to English Words — Hard
Best: Chunk by thousands — O(log n)
"""

class Solution:
    def numberToWords(self, num: int) -> str:
        if num == 0:
            return "Zero"
        below20 = ["","One","Two","Three","Four","Five","Six","Seven","Eight","Nine","Ten",
                   "Eleven","Twelve","Thirteen","Fourteen","Fifteen","Sixteen","Seventeen","Eighteen","Nineteen"]
        tens = ["","","Twenty","Thirty","Forty","Fifty","Sixty","Seventy","Eighty","Ninety"]
        thousands = ["", "Thousand", "Million", "Billion"]

        def three(n):
            if n == 0:
                return ""
            if n < 20:
                return below20[n]
            if n < 100:
                return (tens[n // 10] + (" " + below20[n % 10] if n % 10 else "")).strip()
            return (below20[n // 100] + " Hundred" + (" " + three(n % 100) if n % 100 else "")).strip()

        ans = []
        i = 0
        while num:
            num, rem = divmod(num, 1000)
            if rem:
                chunk = three(rem)
                ans.append(chunk + ((" " + thousands[i]) if thousands[i] else ""))
            i += 1
        return " ".join(reversed(ans))


def _test():
    s = Solution()
    assert s.numberToWords(123) == "One Hundred Twenty Three"
    assert s.numberToWords(1234567) == "One Million Two Hundred Thirty Four Thousand Five Hundred Sixty Seven"
    print("LC 273 OK")


if __name__ == "__main__":
    _test()
