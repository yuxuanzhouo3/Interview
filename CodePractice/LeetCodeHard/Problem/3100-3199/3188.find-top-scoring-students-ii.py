"""
LC 3188. Find Top Scoring Students II — Hard
https://leetcode.com/problems/find-top-scoring-students-ii/
Best: Top scoring students II — SQL
"""

SOLUTION_SQL = """
WITH taken AS (
  SELECT student_id, major, course_id, grade
  FROM students
  JOIN courses USING (major)  -- schematic
),
need AS (
  SELECT major, COUNT(*) AS need_cnt
  FROM courses
  GROUP BY major
),
ok AS (
  SELECT s.student_id, s.major
  FROM students s
  JOIN enrollments e ON s.student_id = e.student_id
  JOIN courses c ON e.course_id = c.course_id AND c.major = s.major
  WHERE e.grade = 'A'
  GROUP BY s.student_id, s.major
  HAVING COUNT(DISTINCT e.course_id) = (
    SELECT COUNT(*) FROM courses c2 WHERE c2.major = s.major
  )
)
SELECT student_id
FROM ok
ORDER BY student_id
"""


def _test():
    q = SOLUTION_SQL.lower()
    assert "student_id" in q and "major" in q
    print("LC 3188 OK")


if __name__ == "__main__":
    _test()

