"""
LC 3832. Find Users with Persistent Behavior Patterns — Hard
https://leetcode.com/problems/find-users-with-persistent-behavior-patterns/
Best: filter single-action days + date-rn grouping for streaks (SQL + pure Python)
"""
from collections import defaultdict
from datetime import datetime, timedelta

SOLUTION_SQL = """
WITH
    daily_counts AS (
        SELECT user_id, action_date, action,
               COUNT(*) OVER (PARTITION BY user_id, action_date) AS cnt
        FROM activity
    ),
    filtered_activity AS (
        SELECT user_id, action_date, action
        FROM daily_counts
        WHERE cnt = 1
    ),
    streak_groups AS (
        SELECT user_id, action, action_date,
               DATE_SUB(action_date, INTERVAL ROW_NUMBER() OVER (
                   PARTITION BY user_id, action ORDER BY action_date
               ) DAY) AS grp
        FROM filtered_activity
    ),
    streak_summary AS (
        SELECT user_id, action, COUNT(*) AS streak_length,
               MIN(action_date) AS start_date, MAX(action_date) AS end_date,
               ROW_NUMBER() OVER (
                   PARTITION BY user_id
                   ORDER BY COUNT(*) DESC, action ASC, MIN(action_date) DESC
               ) AS rnk
        FROM streak_groups
        GROUP BY user_id, action, grp
        HAVING streak_length >= 5
    )
SELECT user_id, action, streak_length, start_date, end_date
FROM streak_summary
WHERE rnk = 1
ORDER BY streak_length DESC, user_id ASC;
"""


def find_behaviorally_stable_users(rows: list[tuple]) -> list[dict]:
    """rows: (user_id, action_date_str, action)"""
    # count actions per (user, date)
    day_cnt = defaultdict(int)
    for u, d, a in rows:
        day_cnt[(u, d)] += 1
    filtered = [(u, d, a) for u, d, a in rows if day_cnt[(u, d)] == 1]
    filtered.sort(key=lambda x: (x[0], x[2], x[1]))
    # group consecutive
    groups = defaultdict(list)  # (user, action, grp) -> dates
    rn = defaultdict(int)
    for u, d, a in filtered:
        rn[(u, a)] += 1
        dt = datetime.strptime(d, "%Y-%m-%d")
        grp = (dt - timedelta(days=rn[(u, a)])).date().isoformat()
        groups[(u, a, grp)].append(d)
    streaks = []
    for (u, a, _), dates in groups.items():
        if len(dates) >= 5:
            streaks.append(
                {
                    "user_id": u,
                    "action": a,
                    "streak_length": len(dates),
                    "start_date": min(dates),
                    "end_date": max(dates),
                }
            )
    # best per user
    best = {}
    for s in streaks:
        u = s["user_id"]
        cur = best.get(u)
        if cur is None or (
            s["streak_length"],
            -ord(s["action"][0]) if False else s["action"],
            s["start_date"],
        ) < (
            # prefer longer, then lex smaller action, then later start
            cur["streak_length"],
            cur["action"],
            # invert start for max: we want later start on tie -> compare negated
            cur["start_date"],
        ):
            # manual compare
            pass
    best = {}
    for s in sorted(
        streaks,
        key=lambda s: (-s["streak_length"], s["action"], s["start_date"]),
        reverse=False,
    ):
        # sort: length desc, action asc, start_date desc
        pass
    streaks.sort(key=lambda s: (-s["streak_length"], s["action"], [-ord(c) for c in s["start_date"]]))
    # start_date desc: use reverse on date string by negating
    streaks.sort(key=lambda s: (-s["streak_length"], s["action"], s["start_date"]))
    # fix start_date to descending: secondary sort
    streaks.sort(key=lambda s: (-s["streak_length"], s["action"], "".join(chr(255 - ord(c)) for c in s["start_date"])))
    for s in streaks:
        if s["user_id"] not in best:
            best[s["user_id"]] = s
    res = sorted(best.values(), key=lambda s: (-s["streak_length"], s["user_id"]))
    return res


class Solution:
    def solve(self, rows):
        return find_behaviorally_stable_users(rows)


def _test():
    rows = [
        (1, "2024-01-01", "login"),
        (1, "2024-01-02", "login"),
        (1, "2024-01-03", "login"),
        (1, "2024-01-04", "login"),
        (1, "2024-01-05", "login"),
        (1, "2024-01-06", "logout"),
        (2, "2024-01-01", "click"),
        (2, "2024-01-02", "click"),
        (2, "2024-01-03", "click"),
        (2, "2024-01-04", "click"),
        (3, "2024-01-01", "view"),
        (3, "2024-01-02", "view"),
        (3, "2024-01-03", "view"),
        (3, "2024-01-04", "view"),
        (3, "2024-01-05", "view"),
        (3, "2024-01-06", "view"),
        (3, "2024-01-07", "view"),
    ]
    out = find_behaviorally_stable_users(rows)
    assert [r["user_id"] for r in out] == [3, 1]
    assert [r["streak_length"] for r in out] == [7, 5]
    assert "WITH" in SOLUTION_SQL and "streak_length" in SOLUTION_SQL
    print("LC 3832 OK")


if __name__ == "__main__":
    _test()
