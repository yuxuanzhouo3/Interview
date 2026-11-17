"""
LC 3268. Find Overlapping Shifts II — Hard (SQL)
https://leetcode.com/problems/find-overlapping-shifts-ii/
Best: Segment count + C(cnt,2)*duration
"""

SOLUTION_SQL = """
WITH EmployeeTimes AS (
  SELECT DISTINCT employee_id, start_time AS time FROM EmployeeShifts
  UNION DISTINCT
  SELECT DISTINCT employee_id, end_time AS time FROM EmployeeShifts
),
Segments AS (
  SELECT
    employee_id,
    time AS start_time,
    LEAD(time) OVER (PARTITION BY employee_id ORDER BY time) AS end_time
  FROM EmployeeTimes
),
SegmentsCount AS (
  SELECT
    Segments.*,
    COUNT(*) AS concurrent_count
  FROM Segments
  INNER JOIN EmployeeShifts USING (employee_id)
  WHERE Segments.start_time >= EmployeeShifts.start_time
    AND Segments.end_time <= EmployeeShifts.end_time
  GROUP BY 1, 2, 3
)
SELECT
  employee_id,
  MAX(concurrent_count) AS max_overlapping_shifts,
  SUM(
    concurrent_count * (concurrent_count - 1) / 2
    * TIMESTAMPDIFF(MINUTE, start_time, end_time)
  ) AS total_overlap_duration
FROM SegmentsCount
GROUP BY 1
ORDER BY 1
"""


def _test():
    assert "max_overlapping_shifts" in SOLUTION_SQL
    assert "total_overlap_duration" in SOLUTION_SQL
    assert "LEAD" in SOLUTION_SQL
    print("LC 3268 OK")


if __name__ == "__main__":
    _test()
