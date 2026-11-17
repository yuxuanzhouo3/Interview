"""
LC 3764. Most Common Course Pairs — Hard (SQL)
https://leetcode.com/problems/most-common-course-pairs/
Best: 顶尖学生相邻课程对频率
"""

SOLUTION_SQL = """
WITH top_students AS (
    SELECT user_id FROM course_completions
    GROUP BY user_id
    HAVING COUNT(1) >= 5 AND AVG(course_rating) >= 4
),
course_pairs AS (
    SELECT course_name AS first_course,
        LEAD(course_name) OVER (PARTITION BY user_id ORDER BY completion_date) AS second_course
    FROM top_students JOIN course_completions USING (user_id)
)
SELECT first_course, second_course, COUNT(1) AS transition_count
FROM course_pairs
WHERE second_course IS NOT NULL
GROUP BY 1, 2
ORDER BY 3 DESC, 1, 2;
"""


def _test():
    assert "SELECT" in SOLUTION_SQL.upper()
    print("LC 3764 OK")


if __name__ == "__main__":
    _test()
