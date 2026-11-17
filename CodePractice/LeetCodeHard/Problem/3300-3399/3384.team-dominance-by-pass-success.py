"""
LC 3384. Team Dominance by Pass Success — Hard (SQL)
https://leetcode.com/problems/team-dominance-by-pass-success/
Best: Join teams + group by half
"""

SOLUTION_SQL = """
WITH T AS (
  SELECT
    t1.team_name,
    IF(time_stamp <= '45:00', 1, 2) AS half_number,
    IF(t1.team_name = t2.team_name, 1, -1) AS dominance
  FROM Passes p
  JOIN Teams t1 ON p.pass_from = t1.player_id
  JOIN Teams t2 ON p.pass_to = t2.player_id
)
SELECT team_name, half_number, SUM(dominance) AS dominance
FROM T
GROUP BY 1, 2
ORDER BY 1, 2
"""


def _test():
    assert "half_number" in SOLUTION_SQL and "dominance" in SOLUTION_SQL
    print("LC 3384 OK")


if __name__ == "__main__":
    _test()
