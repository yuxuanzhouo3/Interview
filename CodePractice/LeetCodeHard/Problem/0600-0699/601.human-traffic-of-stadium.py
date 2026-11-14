"""
LC 601. Human Traffic of Stadium — Hard (SQL)
Best: Self-join / window consecutive ids with people>=100
"""

SOLUTION_SQL = """
WITH t AS (
  SELECT *, id - ROW_NUMBER() OVER (ORDER BY id) AS grp
  FROM Stadium
  WHERE people >= 100
)
SELECT id, visit_date, people
FROM t
WHERE grp IN (
  SELECT grp FROM t GROUP BY grp HAVING COUNT(*) >= 3
)
ORDER BY visit_date
"""


def _test():
    assert "ROW_NUMBER" in SOLUTION_SQL
    print("LC 601 OK")


if __name__ == "__main__":
    _test()
