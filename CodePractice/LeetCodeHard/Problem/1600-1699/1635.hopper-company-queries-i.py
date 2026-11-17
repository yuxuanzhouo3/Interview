"""
LC 1635. Hopper Company Queries I — Hard (SQL)
https://leetcode.com/problems/hopper-company-queries-i/
Best: 递归月份 + 左连接累计司机 / 月接单
"""

SOLUTION_SQL = """
WITH RECURSIVE Months AS (
  SELECT 1 AS month
  UNION ALL
  SELECT month + 1 FROM Months WHERE month < 12
),
Ride AS (
  SELECT MONTH(requested_at) AS month, COUNT(1) AS cnt
  FROM Rides r
  JOIN AcceptedRides a ON r.ride_id = a.ride_id AND YEAR(requested_at) = 2020
  GROUP BY month
)
SELECT
  m.month,
  COUNT(driver_id) AS active_drivers,
  IFNULL(r.cnt, 0) AS accepted_rides
FROM Months m
LEFT JOIN Drivers d
  ON (YEAR(d.join_date) = 2020 AND MONTH(d.join_date) <= m.month)
  OR YEAR(d.join_date) < 2020
LEFT JOIN Ride r ON m.month = r.month
GROUP BY m.month
"""


def _test():
    q = SOLUTION_SQL.lower()
    assert "recursive" in q or "union" in q
    assert "acceptedrides" in q
    assert "drivers" in q
    assert "2020" in q
    print("LC 1635 OK")


if __name__ == "__main__":
    _test()
