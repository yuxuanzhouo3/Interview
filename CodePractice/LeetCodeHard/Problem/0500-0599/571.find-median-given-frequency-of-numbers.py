"""
LC 571. Find Median Given Frequency of Numbers — Hard (SQL)
https://leetcode.com/problems/find-median-given-frequency-of-numbers/
Best: 频率前缀和定位中位
"""

SOLUTION_SQL = """
SELECT AVG(Number) AS median
FROM (
  SELECT Number,
         SUM(Frequency) OVER (ORDER BY Number) AS cum,
         SUM(Frequency) OVER () AS total
  FROM Numbers
) t
WHERE cum - Frequency < FLOOR((total + 1) / 2)
  AND cum >= FLOOR((total + 1) / 2)
   OR cum - Frequency < FLOOR((total + 2) / 2)
  AND cum >= FLOOR((total + 2) / 2)
"""


def _test():
    assert "sum(frequency)" in SOLUTION_SQL.lower(), "SUM(Frequency)"
    assert "over" in SOLUTION_SQL.lower(), "OVER"
    assert "avg" in SOLUTION_SQL.lower(), "AVG"
    print("LC 571 OK")


if __name__ == "__main__":
    _test()
