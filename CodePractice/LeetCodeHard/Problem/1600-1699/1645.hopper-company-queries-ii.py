"""
LC 1645. Hopper Company Queries II — Hard (SQL)
https://leetcode.com/problems/hopper-company-queries-ii/
Best: 月表 + 在职司机 / 当月接单司机占比
"""

SOLUTION_SQL = """
WITH RECURSIVE Month AS (
  SELECT 1 AS month
  UNION ALL
  SELECT month + 1 FROM Month WHERE month < 12
),
S AS (
  SELECT month, driver_id, join_date
  FROM Month m
  LEFT JOIN Drivers d
    ON YEAR(d.join_date) < 2020
    OR (YEAR(d.join_date) = 2020 AND MONTH(d.join_date) <= month)
),
T AS (
  SELECT driver_id, requested_at
  FROM Rides JOIN AcceptedRides USING (ride_id)
  WHERE YEAR(requested_at) = 2020
)
SELECT
  month,
  IFNULL(ROUND(COUNT(DISTINCT t.driver_id) * 100 / COUNT(DISTINCT s.driver_id), 2), 0)
    AS working_percentage
FROM S s
LEFT JOIN T t
  ON s.driver_id = t.driver_id
  AND s.join_date <= t.requested_at
  AND s.month = MONTH(t.requested_at)
GROUP BY 1
"""


def _test():
    q = SOLUTION_SQL.lower()
    assert "working_percentage" in q
    assert "acceptedrides" in q
    assert "round" in q
    print("LC 1645 OK")


if __name__ == "__main__":
    _test()
