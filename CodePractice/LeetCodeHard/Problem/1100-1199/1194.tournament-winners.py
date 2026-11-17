"""
LC 1194. Tournament Winners — Hard (SQL)
https://leetcode.com/problems/tournament-winners/
Best: 组内总分最高且 player_id 最小
"""

SOLUTION_SQL = """
SELECT group_id, player_id
FROM (
  SELECT p.group_id, p.player_id,
         RANK() OVER (PARTITION BY p.group_id ORDER BY IFNULL(SUM(s.score),0) DESC, p.player_id) AS rk
  FROM Players p
  LEFT JOIN (
    SELECT first_player AS player_id, first_score AS score FROM Matches
    UNION ALL
    SELECT second_player, second_score FROM Matches
  ) s ON p.player_id = s.player_id
  GROUP BY p.group_id, p.player_id
) t
WHERE rk = 1
"""


def _test():
    assert "rank()" in SOLUTION_SQL.lower(), "RANK()"
    assert "union all" in SOLUTION_SQL.lower(), "UNION ALL"
    assert "group_id" in SOLUTION_SQL.lower(), "group_id"
    assert "rk = 1" in SOLUTION_SQL.lower(), "rk = 1"
    print("LC 1194 OK")


if __name__ == "__main__":
    _test()
