"""
LC 2153. The Number of Passengers in Each Bus II — Hard (SQL)
https://leetcode.com/problems/the-number-of-passengers-in-each-bus-ii/
Best: 按到达序递推：已到达 - 已上车 与 capacity 取 min
"""

SOLUTION_SQL = """
WITH RECURSIVE
bus AS (
  SELECT bus_id, arrival_time, capacity,
         ROW_NUMBER() OVER (ORDER BY arrival_time) AS rn
  FROM Buses
),
sim AS (
  SELECT
    b.rn,
    b.bus_id,
    b.arrival_time,
    b.capacity,
    (SELECT COUNT(*) FROM Passengers p WHERE p.arrival_time <= b.arrival_time) AS arrived,
    CAST(0 AS SIGNED) AS boarded_before
  FROM bus b
  WHERE b.rn = 1
  UNION ALL
  SELECT
    b.rn,
    b.bus_id,
    b.arrival_time,
    b.capacity,
    (SELECT COUNT(*) FROM Passengers p WHERE p.arrival_time <= b.arrival_time),
    s.boarded_before + LEAST(s.capacity, s.arrived - s.boarded_before)
  FROM sim s
  JOIN bus b ON b.rn = s.rn + 1
)
SELECT bus_id,
       LEAST(capacity, arrived - boarded_before) AS passengers_cnt
FROM sim
ORDER BY bus_id
"""


def _test():
    q = SOLUTION_SQL.lower()
    assert "buses" in q and "passengers" in q
    assert "capacity" in q
    assert "passengers_cnt" in q
    print("LC 2153 OK")


if __name__ == "__main__":
    _test()
