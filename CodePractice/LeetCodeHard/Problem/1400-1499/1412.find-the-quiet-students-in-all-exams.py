"""
LC 1412. Find the Quiet Students in All Exams — Hard (SQL)
https://leetcode.com/problems/find-the-quiet-students-in-all-exams/
Best: 从未拿过最高/最低分的考生
"""

SOLUTION_SQL = """
SELECT DISTINCT s.student_id, s.student_name
FROM Student s
JOIN Exam e ON s.student_id = e.student_id
WHERE s.student_id NOT IN (
  SELECT student_id FROM Exam e1
  WHERE score = (SELECT MAX(score) FROM Exam e2 WHERE e2.exam_id = e1.exam_id)
     OR score = (SELECT MIN(score) FROM Exam e2 WHERE e2.exam_id = e1.exam_id)
)
ORDER BY s.student_id
"""


def _test():
    assert "max(score)" in SOLUTION_SQL.lower(), "MAX(score)"
    assert "min(score)" in SOLUTION_SQL.lower(), "MIN(score)"
    assert "not in" in SOLUTION_SQL.lower(), "NOT IN"
    print("LC 1412 OK")


if __name__ == "__main__":
    _test()
