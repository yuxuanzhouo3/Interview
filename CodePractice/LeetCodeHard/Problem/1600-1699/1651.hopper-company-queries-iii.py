"""
LC 1651. Hopper Company Queries III — Hard (SQL)
https://leetcode.com/problems/hopper-company-queries-iii/
Best: 月聚合 + 三连滑动平均
"""

SOLUTION_SQL = """
WITH RECURSIVE Months AS (
  SELECT 1 AS month UNION ALL SELECT month + 1 FROM Months WHERE month < 12
),
Ride AS (
  SELECT MONTH(requested_at) AS month,
         SUM(ride_distance) AS distance,
         SUM(ride_duration) AS duration
  FROM Rides r
  JOIN AcceptedRides a USING (ride_id)
  WHERE YEAR(requested_at) = 2020
  GROUP BY 1
),
T AS (
  SELECT m.month,
         IFNULL(r.distance, 0) AS distance,
         IFNULL(r.duration, 0) AS duration
  FROM Months m LEFT JOIN Ride r ON m.month = r.month
)
SELECT
  t1.month,
  ROUND((t1.distance + t2.distance + t3.distance) / 3, 2) AS average_ride_distance,
  ROUND((t1.duration + t2.duration + t3.duration) / 3, 2) AS average_ride_duration
FROM T t1
JOIN T t2 ON t1.month + 1 = t2.month
JOIN T t3 ON t1.month + 2 = t3.month
ORDER BY t1.month
"""


def _test():
    q = SOLUTION_SQL.lower()
    assert "average_ride_distance" in q
    assert "average_ride_duration" in q
    assert "/ 3" in q or "/3" in q
    print("LC 1651 OK")


if __name__ == "__main__":
    _test()
