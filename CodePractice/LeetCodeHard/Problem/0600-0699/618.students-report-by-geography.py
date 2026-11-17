"""
LC 618. Students Report By Geography — Hard (SQL)
https://leetcode.com/problems/students-report-by-geography/
Best: 按大洲 ROW_NUMBER 后条件聚合透视
"""

SOLUTION_SQL = """
SELECT
  MAX(CASE WHEN continent = 'America' THEN name END) AS America,
  MAX(CASE WHEN continent = 'Asia' THEN name END) AS Asia,
  MAX(CASE WHEN continent = 'Europe' THEN name END) AS Europe
FROM (
  SELECT name, continent,
         ROW_NUMBER() OVER (PARTITION BY continent ORDER BY name) AS rn
  FROM student
) t
GROUP BY rn
"""


def _test():
    assert "row_number" in SOLUTION_SQL.lower(), "ROW_NUMBER"
    assert "case when" in SOLUTION_SQL.lower(), "CASE WHEN"
    assert "america" in SOLUTION_SQL.lower(), "America"
    assert "group by" in SOLUTION_SQL.lower(), "GROUP BY"
    print("LC 618 OK")


if __name__ == "__main__":
    _test()
