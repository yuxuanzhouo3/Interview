#!/usr/bin/env python3
"""Generate problem.md from _hard_index.json (LeetCode topic classify)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent

CATEGORY_ORDER = [
    "Array",
    "String",
    "Hash Table",
    "Dynamic Programming",
    "Math",
    "Sorting",
    "Greedy",
    "Depth-First Search",
    "Breadth-First Search",
    "Binary Search",
    "Tree",
    "Binary Tree",
    "Matrix",
    "Two Pointers",
    "Bit Manipulation",
    "Stack",
    "Heap (Priority Queue)",
    "Prefix Sum",
    "Simulation",
    "Counting",
    "Design",
    "Graph",
    "Graph Theory",
    "Sliding Window",
    "Backtracking",
    "Union-Find",
    "Linked List",
    "Ordered Set",
    "Monotonic Stack",
    "Monotonic Queue",
    "Enumeration",
    "Recursion",
    "Trie",
    "Divide and Conquer",
    "Binary Search Tree",
    "Bitmask",
    "Queue",
    "Memoization",
    "Geometry",
    "Segment Tree",
    "Topological Sort",
    "Number Theory",
    "Hash Function",
    "Binary Indexed Tree",
    "Game Theory",
    "Data Stream",
    "Interactive",
    "String Matching",
    "Rolling Hash",
    "Shortest Path",
    "Combinatorics",
    "Randomized",
    "Brainteaser",
    "Merge Sort",
    "Iterator",
    "Concurrency",
    "Doubly-Linked List",
    "Probability and Statistics",
    "Quickselect",
    "Bucket Sort",
    "Suffix Array",
    "Minimum Spanning Tree",
    "Counting Sort",
    "Shell",
    "Line Sweep",
    "Reservoir Sampling",
    "Eulerian Circuit",
    "Radix Sort",
    "Strongly Connected Component",
    "Rejection Sampling",
    "Biconnected Component",
    "Database",
    "JavaScript",
    "Pandas",
    "Untagged",
]


def slugify(name: str) -> str:
    s = name.lower()
    for a, b in [
        (" ", "-"),
        ("(", ""),
        (")", ""),
        ("/", ""),
        (",", ""),
    ]:
        s = s.replace(a, b)
    return s


def bucket(pid: int) -> str:
    return f"{(pid // 100) * 100:04d}-{((pid // 100) * 100 + 99):04d}"


def is_filled(root: Path, pid: int, slug: str) -> bool:
    py = root / bucket(pid) / f"{pid}.{slug}.py"
    if not py.exists():
        return False
    text = py.read_text(encoding="utf-8")
    return "Status: STUB" not in text and "raise NotImplementedError" not in text


def main() -> None:
    with open(ROOT / "_hard_index.json", encoding="utf-8") as f:
        problems = json.load(f)

    by_id = {p["id"]: p for p in problems}
    tag_ids: dict[str, list[int]] = defaultdict(list)
    for p in problems:
        tags = p["tagNames"] or ["Untagged"]
        for t in tags:
            tag_ids[t].append(p["id"])

    filled_set = {i for i, p in by_id.items() if is_filled(ROOT, i, p["slug"])}
    filled_ids = sorted(filled_set)

    ordered: list[str] = []
    seen: set[str] = set()
    for t in CATEGORY_ORDER:
        if t in tag_ids:
            ordered.append(t)
            seen.add(t)
    rest = sorted(
        (t for t in tag_ids if t not in seen),
        key=lambda t: (-len(tag_ids[t]), t),
    )
    ordered.extend(rest)

    def link_id(pid: int) -> str:
        p = by_id[pid]
        mark = "✓" if pid in filled_set else ""
        return f"[{pid}{mark}]({bucket(pid)}/{pid}.{p['slug']}.md)"

    lines: list[str] = []
    lines.append("# LeetCode Hard · Problem Classify")
    lines.append("")
    lines.append(
        "来源：LeetCode GraphQL `difficulty=Hard`，`frontendQuestionId ∈ [1, 5000]`。"
    )
    lines.append("分类按官方 **Topic Tags**；一题可属多类。")
    lines.append("")
    lines.append("## 规模")
    lines.append("")
    lines.append("| 项 | 值 |")
    lines.append("|----|----|")
    lines.append(f"| Hard 总数 | **{len(problems)}** |")
    lines.append(
        f"| 免费 / Premium | **{sum(1 for p in problems if not p['paidOnly'])}** / "
        f"**{sum(1 for p in problems if p['paidOnly'])}** |"
    )
    lines.append(f"| ID 范围 | **{problems[0]['id']} – {problems[-1]['id']}** |")
    lines.append(f"| Tag 种类 | **{len(tag_ids)}** |")
    lines.append(f"| 已填完整解 ✓ | **{len(filled_ids)}** |")
    lines.append("")
    lines.append("## 约定")
    lines.append("")
    lines.append("- 文件：`{id}.{slug}.py` + `{id}.{slug}.md`（百位分桶）")
    lines.append("- 元数据：`_hard_index.json`")
    lines.append("- 表内 ID 链到题解 md；后缀 **✓** = 已写完整最优解 + 可跑测")
    lines.append("- 本文件替代原 `README.md`")
    lines.append("")
    lines.append("## 题型总览")
    lines.append("")
    lines.append("| Type | Count | Problem IDs |")
    lines.append("|------|------:|-------------|")

    for t in ordered:
        ids = sorted(tag_ids[t])
        id_str = ", ".join(str(i) for i in ids)
        lines.append(f"| [{t}](#{slugify(t)}) | {len(ids)} | {id_str} |")

    lines.append("")
    lines.append("## 各题型明细")
    lines.append("")

    for t in ordered:
        ids = sorted(tag_ids[t])
        lines.append(f"### {t}")
        lines.append("")
        lines.append(f"共 **{len(ids)}** 题。")
        lines.append("")
        lines.append("| ID | Title | Premium | Other Tags |")
        lines.append("|----|-------|---------|------------|")
        for pid in ids:
            p = by_id[pid]
            title = f"[{p['title']}]({bucket(pid)}/{pid}.{p['slug']}.md)"
            if pid in filled_set:
                title = f"{title} ✓"
            prem = "Yes" if p["paidOnly"] else ""
            others = ", ".join(x for x in p["tagNames"] if x != t) or "—"
            lines.append(f"| {link_id(pid)} | {title} | {prem} | {others} |")
        lines.append("")

    lines.append("## 按 ID 全表")
    lines.append("")
    lines.append("| ID | Title | Premium | Tags |")
    lines.append("|----|-------|---------|------|")
    for p in problems:
        pid = p["id"]
        title = f"[{p['title']}]({bucket(pid)}/{pid}.{p['slug']}.md)"
        if pid in filled_set:
            title += " ✓"
        prem = "Yes" if p["paidOnly"] else ""
        tags = ", ".join(p["tagNames"]) or "—"
        lines.append(f"| {link_id(pid)} | {title} | {prem} | {tags} |")
    lines.append("")

    out = ROOT / "problem.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    readme = ROOT / "README.md"
    if readme.exists():
        readme.unlink()
    print(f"wrote {out} lines={len(lines)} filled={len(filled_ids)} tags={len(ordered)}")


if __name__ == "__main__":
    main()
