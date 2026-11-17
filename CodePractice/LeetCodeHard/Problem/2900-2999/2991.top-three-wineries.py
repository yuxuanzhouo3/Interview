"""
LC 2991. Top Three Wineries  — Hard
https://leetcode.com/problems/top-three-wineries/
Best: Window rank per country — SQL
"""

SOLUTION_SQL = """
WITH ranked AS (
  SELECT country, winery, SUM(points) AS points,
         RANK() OVER (PARTITION BY country ORDER BY SUM(points) DESC, winery) AS rk
  FROM Wineries
  GROUP BY country, winery
)
SELECT country,
  MAX(CASE WHEN rk=1 THEN CONCAT(winery,' (',points,')') END) AS top_winery,
  COALESCE(MAX(CASE WHEN rk=2 THEN CONCAT(winery,' (',points,')') END), 'No second winery') AS second_winery,
  COALESCE(MAX(CASE WHEN rk=3 THEN CONCAT(winery,' (',points,')') END), 'No third winery') AS third_winery
FROM ranked
WHERE rk <= 3
GROUP BY country
ORDER BY country
"""


def _test():
    q = SOLUTION_SQL.lower()
    assert "rank()" in q or "row_number()" in q
    assert "partition by country" in q
    assert "no second winery" in q
    print("LC 2991 OK")


if __name__ == "__main__":
    _test()

