"""
LC 2759. Convert JSON String to Object — Hard (JS)
Best: Recursive descent parser — O(n)
"""

class Solution:
    def jsonParse(self, str_: str):
        s = str_
        i = 0

        def peek():
            nonlocal i
            while i < len(s) and s[i].isspace():
                i += 1
            return s[i] if i < len(s) else ""

        def parse():
            nonlocal i
            c = peek()
            if c == "{":
                return parse_obj()
            if c == "[":
                return parse_arr()
            if c == '"':
                return parse_str()
            if c == "t":
                i += 4
                return True
            if c == "f":
                i += 5
                return False
            if c == "n":
                i += 4
                return None
            return parse_num()

        def parse_str():
            nonlocal i
            i += 1
            out = []
            while s[i] != '"':
                if s[i] == "\\":
                    i += 1
                    out.append(s[i])
                else:
                    out.append(s[i])
                i += 1
            i += 1
            return "".join(out)

        def parse_num():
            nonlocal i
            j = i
            if s[i] == "-":
                i += 1
            while i < len(s) and (s[i].isdigit() or s[i] in ".eE+-"):
                i += 1
            num = s[j:i]
            return float(num) if any(c in num for c in ".eE") else int(num)

        def parse_arr():
            nonlocal i
            i += 1
            arr = []
            if peek() == "]":
                i += 1
                return arr
            while True:
                arr.append(parse())
                if peek() == ",":
                    i += 1
                    continue
                i += 1  # ]
                return arr

        def parse_obj():
            nonlocal i
            i += 1
            obj = {}
            if peek() == "}":
                i += 1
                return obj
            while True:
                key = parse_str()
                peek()
                i += 1  # :
                obj[key] = parse()
                if peek() == ",":
                    i += 1
                    continue
                i += 1
                return obj

        return parse()


def _test():
    s = Solution()
    assert s.jsonParse('{"a":1}') == {"a": 1}
    assert s.jsonParse("[1,2,true]") == [1, 2, True]
    print("LC 2759 OK")


if __name__ == "__main__":
    _test()
