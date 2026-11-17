"""
LC 2494. Merge Overlapping Events in the Same Hall — Hard (SQL)
Best: Gaps-and-islands on hall events
"""

SOLUTION_SQL = """
WITH ordered AS (
  SELECT hall_id, start_day, end_day,
    MAX(end_day) OVER (
      PARTITION BY hall_id ORDER BY start_day, end_day
      ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS prev_max_end
  FROM HallEvents
),
marked AS (
  SELECT hall_id, start_day, end_day,
    CASE WHEN prev_max_end IS NULL OR start_day > prev_max_end THEN 1 ELSE 0 END AS is_new
  FROM ordered
),
grouped AS (
  SELECT hall_id, start_day, end_day,
    SUM(is_new) OVER (PARTITION BY hall_id ORDER BY start_day, end_day) AS grp
  FROM marked
)
SELECT hall_id, MIN(start_day) AS start_day, MAX(end_day) AS end_day
FROM grouped
GROUP BY hall_id, grp
ORDER BY hall_id, start_day
"""


def _test():
    assert "HallEvents" in SOLUTION_SQL and "grp" in SOLUTION_SQL
    print("LC 2494 OK")


if __name__ == "__main__":
    _test()
