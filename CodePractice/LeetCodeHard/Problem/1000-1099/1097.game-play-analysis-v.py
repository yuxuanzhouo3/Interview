"""
LC 1097. Game Play Analysis V — Hard (SQL)
https://leetcode.com/problems/game-play-analysis-v/
Best: 首登日 LEFT JOIN 次日
"""

SOLUTION_SQL = """
SELECT
  a.install_dt,
  COUNT(a.player_id) AS installs,
  ROUND(COUNT(b.player_id) / COUNT(a.player_id), 2) AS Day1_retention
FROM (
  SELECT player_id, MIN(event_date) AS install_dt
  FROM Activity
  GROUP BY player_id
) a
LEFT JOIN Activity b
  ON a.player_id = b.player_id
 AND b.event_date = DATE_ADD(a.install_dt, INTERVAL 1 DAY)
GROUP BY a.install_dt
"""


def _test():
    assert "install_dt" in SOLUTION_SQL.lower(), "install_dt"
    assert "day1_retention" in SOLUTION_SQL.lower(), "Day1_retention"
    assert "date_add" in SOLUTION_SQL.lower(), "DATE_ADD"
    assert "left join" in SOLUTION_SQL.lower(), "LEFT JOIN"
    print("LC 1097 OK")


if __name__ == "__main__":
    _test()
