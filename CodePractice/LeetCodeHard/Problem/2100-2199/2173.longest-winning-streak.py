"""
LC 2173. Longest Winning Streak — Hard (SQL)
https://leetcode.com/problems/longest-winning-streak/
Best: 胜场连续段最大长度
"""

SOLUTION_SQL = """
WITH ordered AS (
  SELECT player_id, result, match_day,
         ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY match_day)
         - ROW_NUMBER() OVER (PARTITION BY player_id, result ORDER BY match_day) AS grp
  FROM Matches
)
SELECT player_id, MAX(streak) AS longest_streak
FROM (
  SELECT player_id, COUNT(*) AS streak
  FROM ordered
  WHERE result = 'Win'
  GROUP BY player_id, grp
) t
GROUP BY player_id
"""


def _test():
    assert "row_number" in SOLUTION_SQL.lower(), "ROW_NUMBER"
    assert "win" in SOLUTION_SQL.lower(), "Win"
    assert "longest_streak" in SOLUTION_SQL.lower(), "longest_streak"
    assert "grp" in SOLUTION_SQL.lower(), "grp"
    print("LC 2173 OK")


if __name__ == "__main__":
    _test()
