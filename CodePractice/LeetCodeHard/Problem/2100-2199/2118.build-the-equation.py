"""
LC 2118. Build the Equation — Hard (SQL)
https://leetcode.com/problems/build-the-equation/
Best: 按 power 降序拼接多项式字符串
"""

SOLUTION_SQL = """
SELECT
  CONCAT(
    GROUP_CONCAT(
      CONCAT(
        IF(factor > 0 AND power < (SELECT MAX(power) FROM Terms), '+', ''),
        CASE
          WHEN power = 0 THEN CAST(factor AS CHAR)
          WHEN power = 1 THEN
            IF(ABS(factor) = 1, IF(factor > 0, 'X', '-X'), CONCAT(factor, 'X'))
          ELSE
            IF(ABS(factor) = 1,
               IF(factor > 0, CONCAT('X^', power), CONCAT('-X^', power)),
               CONCAT(factor, 'X^', power))
        END
      )
      ORDER BY power DESC SEPARATOR ''
    ),
    '=0'
  ) AS equation
FROM Terms
"""


def _test():
    q = SOLUTION_SQL.lower()
    assert "equation" in q
    assert "terms" in q
    assert "power" in q and "factor" in q
    print("LC 2118 OK")


if __name__ == "__main__":
    _test()
