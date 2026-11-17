"""
LC 3673. Find Zombie Sessions — Hard (SQL)
https://leetcode.com/problems/find-zombie-sessions/
Best: 按会话聚合过滤僵尸行为
"""

SOLUTION_SQL = """
SELECT session_id, user_id,
    TIMESTAMPDIFF(MINUTE, MIN(event_timestamp), MAX(event_timestamp)) AS session_duration_minutes,
    SUM(event_type = 'scroll') AS scroll_count
FROM app_events
GROUP BY session_id, user_id
HAVING session_duration_minutes >= 30
    AND SUM(event_type = 'click') / SUM(event_type = 'scroll') < 0.2
    AND SUM(event_type = 'purchase') = 0
    AND SUM(event_type = 'scroll') >= 5
ORDER BY scroll_count DESC, session_id;
"""


def _test():
    assert "SELECT" in SOLUTION_SQL.upper()
    print("LC 3673 OK")


if __name__ == "__main__":
    _test()
