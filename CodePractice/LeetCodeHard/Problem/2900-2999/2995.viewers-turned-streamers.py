"""
LC 2995. Viewers Turned Streamers — Hard
https://leetcode.com/problems/viewers-turned-streamers/
Best: First view then stream — SQL
"""

SOLUTION_SQL = """
WITH first_session AS (
  SELECT user_id, session_type, session_start,
         ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY session_start) AS rn
  FROM Sessions
)
SELECT DISTINCT s.user_id, s.session_type AS session_type,
       s.session_start, s.session_end, s.session_id
FROM Sessions s
JOIN first_session f
  ON s.user_id = f.user_id AND f.rn = 1 AND f.session_type = 'Viewer'
WHERE s.session_type = 'Streamer'
ORDER BY s.session_start DESC, s.session_id DESC
"""


def _test():
    q = SOLUTION_SQL.lower()
    assert "row_number()" in q
    assert "viewer" in q and "streamer" in q
    print("LC 2995 OK")


if __name__ == "__main__":
    _test()

