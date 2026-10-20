"""Deterministic reconciliation checks for a document index (one row per document).

Usage: python3 checks/reconcile.py [path/to/book.xlsx|.csv] [--out DIR]

Assumptions (state these wherever results are shown):
  - "contributed" is gross: total capital paid in, never reduced by distributions.
  - A capital call counts toward a statement if its due date (effective_date) is on or before the
    statement's as-of date. A distribution counts if its effective_date is on or before the as-of date.
  - Money comparisons use a $1 tolerance (some managers round statement figures to whole dollars).
  - Tax estimates are issued before the year they estimate ends, so they are exempt from the
    "posted before its own date" rule.
  - Being stale (no statement for the latest quarter) is reported as information, not as a failure.
"""
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

POS = ["client", "investing_entity", "fund"]
MONEY_TYPES = ("Account Statement", "Capital Call", "Distribution Notice")
TOL = 1.0
JUMP = 5.0  # a quarter-over-quarter value move of >= 5x (or <= 1/5x) is suspicious
LATEST_QUARTER = pd.Timestamp("2024-12-31")

CHECKS = {
    1: "Money paid in",
    2: "Money paid back",
    3: "Sudden jumps",
    4: "Duplicates",
    5: "Wrong label",
    6: "Wrong date",
    7: "Missing pieces",
    8: "Same fund, same share",
}
BLOCKING = "blocking"  # changes a number the client sees
HYGIENE = "hygiene"  # document list quality; does not change client numbers


@dataclass
class Finding:
    check: int
    rows: list
    position: tuple
    severity: str
    issue: str
    related: list = field(default_factory=list)


def load(path):
    path = Path(path)
    d = pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path)
    for c in ("effective_date", "posted_date"):
        d[c] = pd.to_datetime(d[c])
    d["position"] = list(zip(*(d[c] for c in POS)))
    return d


def _money(x):
    s = f"${abs(x):,.2f}".removesuffix(".00")
    return f"-{s}" if x < 0 else s


def normalize_name(name):
    """File name with download-copy suffixes "(2) (3)" and a repeated trailing " - <prefix>" removed."""
    n = re.sub(r"\s*\(\d+\)", "", name).strip()
    m = re.match(r"^(.*)\s+-\s+([^-]+)$", n)
    if m and m.group(2).strip().lower() in m.group(1).lower():
        n = m.group(1)
    return re.sub(r"[\s._\-,]+", " ", n).strip().lower()


# ---------- check 4: duplicates (runs first so roll-forwards don't double count)
def check_duplicates(d):
    out = []
    cash = d[d.cash_amount.notna()]
    key = ["position", "document_type", "effective_date", "cash_amount"]
    for _, g in cash[cash.duplicated(key, keep=False)].groupby(key):
        rows = sorted(g.row_id)
        out.append(Finding(4, rows, g.position.iloc[0], BLOCKING,
                           f"Same {g.document_type.iloc[0].lower()} of {_money(abs(g.cash_amount.iloc[0]))} "
                           f"due {g.effective_date.iloc[0].date()} appears {len(rows)} times"))
    docs = d[d.cash_amount.isna()].assign(norm=lambda x: x.file_name.map(normalize_name))
    key = ["position", "document_type", "effective_date", "norm"]
    for _, g in docs[docs.duplicated(key, keep=False)].groupby(key):
        rows = sorted(g.row_id)
        out.append(Finding(4, rows, g.position.iloc[0], HYGIENE,
                           f"Same document stored {len(rows)} times: {g.file_name.iloc[0]}"))
    return out


def duplicate_rows_to_drop(findings):
    """For blocking (cash) duplicates keep the lowest row_id and drop the rest."""
    return {r for f in findings if f.check == 4 and f.severity == BLOCKING for r in f.rows[1:]}


# ---------- checks 1 & 2: roll-forwards
def _roll_forward(d, drop, check, doc_type, field_name, noun):
    out = []
    stmts = d[(d.document_type == "Account Statement") & d[field_name].notna()]
    src = d[(d.document_type == doc_type) & ~d.row_id.isin(drop)]
    for _, s in stmts.iterrows():
        mine = src[src.position == s.position].sort_values("effective_date")
        upto = mine[mine.effective_date <= s.effective_date]
        after = mine[mine.effective_date > s.effective_date]
        expected = upto.cash_amount.abs().sum()
        diff = s[field_name] - expected
        if abs(diff) <= TOL:
            continue
        cum = after.cash_amount.abs().cumsum()
        hit = [i for i, v in enumerate(cum) if abs(v - diff) <= TOL]
        if hit:
            later = list(after.row_id[: hit[0] + 1])
            why = f"includes {noun}s dated after {s.effective_date.date()} (rows {', '.join(map(str, later))})"
        elif len(upto) == 0:
            later, why = [], f"no {noun} notices on file up to this date"
        else:
            later, why = [], "no combination of notices explains the difference"
        out.append(Finding(check, [s.row_id], s.position, BLOCKING,
                           f"Statement {_money(s[field_name])} vs notices {_money(expected)} "
                           f"(off by {_money(diff)}); {why}", related=later))
    return out


def check_contributed(d, drop):
    return _roll_forward(d, drop, 1, "Capital Call", "contributed", "call")


def check_distributions(d, drop):
    return _roll_forward(d, drop, 2, "Distribution Notice", "cumulative_distributions", "distribution")


# ---------- check 3: period-over-period sanity
def check_sanity(d):
    out = []
    stmts = d[(d.document_type == "Account Statement") & d.contributed.notna()]
    for pos, g in stmts.sort_values("effective_date").groupby("position", sort=False):
        g = g.reset_index(drop=True)
        flagged = set()
        for i, r in g.iterrows():
            if r.contributed > r.commitment + TOL:
                out.append(Finding(3, [r.row_id], pos, BLOCKING, "Paid in more than the commitment"))
            if r.cumulative_distributions > 0 and abs(r.total_value - r.cumulative_distributions) <= TOL:
                flagged.add(r.row_id)
                out.append(Finding(3, [r.row_id], pos, BLOCKING,
                                   f"Value equals cumulative distributions ({_money(r.total_value)}); "
                                   "likely read from the wrong column"))
            if i == 0:
                continue
            p = g.iloc[i - 1]
            if r.commitment != p.commitment:
                out.append(Finding(3, [r.row_id], pos, BLOCKING,
                                   f"Commitment changed {_money(p.commitment)} -> {_money(r.commitment)}"))
            if r.contributed < p.contributed - TOL:
                out.append(Finding(3, [r.row_id], pos, BLOCKING,
                                   f"Paid-in fell {_money(p.contributed)} -> {_money(r.contributed)}"))
            if r.cumulative_distributions < p.cumulative_distributions - TOL:
                out.append(Finding(3, [r.row_id], pos, BLOCKING, "Cumulative distributions fell"))
        # value jumps: a crash followed by a recovery flags the rows in between (the bad points)
        vals = g[g.total_value > 0]
        moves = [(j, vals.total_value.iloc[j] / vals.total_value.iloc[j - 1]) for j in range(1, len(vals))]
        jumps = [(j, r) for j, r in moves if r >= JUMP or r <= 1 / JUMP]
        used = set()
        for a, (j, r) in enumerate(jumps):
            if j in used:
                continue
            back = next(((k, r2) for k, r2 in jumps[a + 1:] if (r2 > 1) != (r > 1)), None)
            if back:
                rows = list(vals.row_id.iloc[j:back[0]])
                used.add(back[0])
                out.append(Finding(3, rows, pos, BLOCKING,
                                   f"Value moved {r:.2f}x then {back[1]:.2f}x back; the rows in between "
                                   "are likely mis-read"))
            elif vals.row_id.iloc[j - 1] not in flagged:
                out.append(Finding(3, [vals.row_id.iloc[j]], pos, BLOCKING,
                                   f"Value moved {r:.2f}x in one quarter "
                                   f"(vs row {vals.row_id.iloc[j - 1]})"))
    return out


# ---------- check 5: classification
RULES = [
    (r"pcap|summary report|financial summary|accountstatement", "Account Statement"),
    (r"audited|\bafs\b|auditedfs", "Financial Statements"),
    (r"k-?1|tax", "Tax"),
    (r"cc_notice|call notice|capital call|drawdown|^notice_\d|additional payment", "Capital Call"),
    (r"^distribution|distribution (on|#)", "Distribution Notice"),
    (r"lpa|ppm|agreement|call rights|ddq|privacy|policy", "Legal"),
    (r"report|quarterly fs|_fs_|_fs$|acquisition|dividend recap|portco", "Investment Update"),
]


def expected_type(file_name):
    n = file_name.lower()
    return next((t for p, t in RULES if re.search(p, n)), None)


def check_classification(d):
    out = []
    for _, r in d.iterrows():
        exp = expected_type(r.file_name)
        if exp and exp != r.document_type:
            out.append(Finding(5, [r.row_id], r.position, HYGIENE,
                               f"Classified as {r.document_type}; file name says {exp}"))
        if r.document_type == "Account Statement" and pd.isna(r.contributed):
            out.append(Finding(5, [r.row_id], r.position, HYGIENE, "Account Statement with no figures"))
        if r.document_type not in MONEY_TYPES and r[["commitment", "contributed", "total_value",
                                                     "cumulative_distributions", "cash_amount"]].notna().any():
            out.append(Finding(5, [r.row_id], r.position, HYGIENE,
                               f"{r.document_type} carries money figures"))
    return out


# ---------- check 6: date integrity
MON = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}


def _ts(y, m, d):
    try:
        return pd.Timestamp(y, m, d)
    except ValueError:
        return None


def file_dates(name):
    """Calendar dates in a file name (both US and day-first readings) and quarter-end labels."""
    dates, quarters = [], []
    for a, b, c in re.findall(r"(?<!\d)(\d{1,2})[.\-](\d{1,2})[.\-](\d{2,4})(?!\d)", name):
        y = int(c) + (2000 if len(c) == 2 else 0)
        dates += [t for t in (_ts(y, int(a), int(b)), _ts(y, int(b), int(a))) if t is not None]
    for mo, dd, y in re.findall(r"(?i)\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.? (\d{1,2}),? (\d{4})", name):
        dates.append(_ts(int(y), MON[mo.lower()], int(dd)))
    for dd, mo, y in re.findall(r"(?i)\b(\d{1,2}) (jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* (\d{4})", name):
        dates.append(_ts(int(y), MON[mo.lower()], int(dd)))
    for q, y in re.findall(r"(?i)q([1-4])[ _\-]?(\d{2}(?:\d{2})?)(?!\d)", name):
        y = int(y) + (2000 if len(y) == 2 else 0)
        quarters.append(pd.Timestamp(y, int(q) * 3, 1) + pd.offsets.MonthEnd(0))
    return [x for x in dates if x is not None], quarters


def check_dates(d):
    out = []
    for _, r in d.iterrows():
        f = lambda issue: out.append(Finding(6, [r.row_id], r.position, HYGIENE, issue))
        dates, quarters = file_dates(r.file_name)
        if dates and not any(x in (r.effective_date, r.posted_date) for x in dates):
            f(f"File name date {dates[0].date()} matches neither effective {r.effective_date.date()} "
              f"nor posted {r.posted_date.date()}")
        elif quarters and not dates and r.effective_date not in quarters:
            f(f"Quarter label {quarters[0].date()} but effective date {r.effective_date.date()}")
        years = re.findall(r"(?<!\d)(20\d\d)(?!\d)", r.file_name)
        if (not dates and not quarters and years and r.document_type in ("Tax", "Financial Statements")
                and str(r.effective_date.year) not in years):
            f(f"Year label {years[0]} but effective date {r.effective_date.date()}")
        estimate = r.document_type == "Tax" and "estimat" in r.file_name.lower()
        if r.document_type != "Capital Call" and not estimate and r.posted_date < r.effective_date:
            f(f"Posted {r.posted_date.date()}, before its own date {r.effective_date.date()}")
        if r.document_type == "Capital Call" and r.effective_date < r.posted_date:
            f("Call due before it was issued")
    return out


# ---------- check 7: completeness
def check_completeness(d, drop):
    out = []
    for _, r in d[d.portal.isna()].iterrows():
        out.append(Finding(7, [r.row_id], r.position, HYGIENE, "No source portal; origin cannot be verified"))
    calls = d[(d.document_type == "Capital Call") & ~d.row_id.isin(drop)]
    for pos, g in d.groupby("position"):
        stmts = g[(g.document_type == "Account Statement") & g.contributed.notna()].sort_values("effective_date")
        if stmts.empty:
            continue
        cash = g[g.document_type.isin(["Capital Call", "Distribution Notice"])]
        first = min(stmts.effective_date.min(), cash.effective_date.min() if len(cash) else pd.NaT)
        last = stmts.effective_date.max()
        have = set(stmts.effective_date)
        gaps = [q.date() for q in pd.date_range(first + pd.offsets.QuarterEnd(0), last, freq="QE")
                if q not in have]
        if gaps:
            out.append(Finding(7, list(stmts.row_id[:1]), pos, HYGIENE,
                               f"Quarterly statements missing for {', '.join(map(str, gaps))}"))
        s0 = stmts.iloc[0]
        if s0.contributed > 0 and not len(calls[(calls.position == pos) & (calls.effective_date <= s0.effective_date)]):
            out.append(Finding(7, [s0.row_id], pos, HYGIENE,
                               f"First statement shows {_money(s0.contributed)} paid in, but no earlier calls on file"))
    return out


def staleness(d):
    stmts = d[(d.document_type == "Account Statement") & d.contributed.notna()]
    latest = stmts.groupby("position").effective_date.max()
    return {pos: dt.date() for pos, dt in latest.items() if dt < LATEST_QUARTER}


# ---------- check 8: co-investor pro-rata
def check_pro_rata(d, drop):
    out = []
    commit = (d[d.commitment.notna()].sort_values("effective_date")
              .groupby("position").commitment.last())
    cash = d[d.document_type.isin(["Capital Call", "Distribution Notice"]) & ~d.row_id.isin(drop)].copy()
    cash = cash[cash.position.isin(commit.index)]
    cash["pct"] = cash.cash_amount.abs() / cash.position.map(commit) * 100
    for fund, f in cash.groupby("fund"):
        holders = {p for p in commit.index if p[2] == fund}
        if len(holders) < 2:
            continue
        for (doc_type, when), g in f.groupby(["document_type", "effective_date"]):
            noun = "call" if doc_type == "Capital Call" else "distribution"
            if g.pct.max() - g.pct.min() > 0.01:
                shares = ", ".join(f"{p[1]} {v:.2f}%" for p, v in zip(g.position, g.pct))
                for _, r in g.iterrows():
                    out.append(Finding(8, [r.row_id], r.position, BLOCKING,
                                       f"{when.date()} {noun}: co-investors received different shares of "
                                       f"commitment ({shares})",
                                       related=[x for x in g.row_id if x != r.row_id]))
            for pos in holders - set(g.position):
                out.append(Finding(8, [], pos, BLOCKING,
                                   f"Co-investors have a {noun} on {when.date()} "
                                   f"(rows {', '.join(map(str, g.row_id))}) but this position has none",
                                   related=list(g.row_id)))
    return out


# ---------- run all
def run(d):
    dups = check_duplicates(d)
    drop = duplicate_rows_to_drop(dups)
    findings = (check_contributed(d, drop) + check_distributions(d, drop) + check_sanity(d) + dups
                + check_classification(d) + check_dates(d) + check_completeness(d, drop)
                + check_pro_rata(d, drop))
    return sorted(findings, key=lambda f: (f.check, f.position, f.rows))


def grid(d, findings):
    """Position x check: 'blocking', 'hygiene' or 'pass' (worst severity wins)."""
    positions = sorted(set(d.position))
    g = pd.DataFrame("pass", index=pd.MultiIndex.from_tuples(positions, names=POS), columns=list(CHECKS))
    for f in findings:
        if g.loc[f.position, f.check] != BLOCKING:
            g.loc[f.position, f.check] = f.severity
    return g


def reconciled_positions(g):
    return [pos for pos, row in g.iterrows() if BLOCKING not in row.values]


def main(argv):
    out_dir = None
    if "--out" in argv:
        i = argv.index("--out")
        out_dir = Path(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    path = argv[0] if argv else Path(__file__).resolve().parent.parent / "harbor_ridge_book_3552.xlsx"
    d = load(path)
    findings = run(d)
    g = grid(d, findings)

    for n, name in CHECKS.items():
        fs = [f for f in findings if f.check == n]
        print(f"\n== {n}. {name} ({len(fs)} findings)")
        for f in fs:
            rows = ", ".join(map(str, f.rows)) or "-"
            print(f"  [{f.severity[0].upper()}] rows {rows} | {f.position[1]} / {f.position[2][:30]} | {f.issue}")

    sym = {BLOCKING: "X", HYGIENE: "~", "pass": "."}
    print("\n== Grid (X = changes a client number, ~ = document hygiene, . = pass)")
    print(f"{'entity / fund':62}" + " ".join(str(c) for c in CHECKS))
    for pos, row in g.iterrows():
        print(f"{(pos[1] + ' / ' + pos[2])[:61]:62}" + " ".join(sym[v] for v in row))
    ok = reconciled_positions(g)
    print(f"\n{len(ok)} of {len(g)} positions reconcile: " + "; ".join(f"{p[1]} / {p[2]}" for p in ok))
    stale = staleness(d)
    print(f"{len(stale)} positions have no statement for {LATEST_QUARTER.date()} (shown with their as-of date)")

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"check": f.check, "check_name": CHECKS[f.check], "severity": f.severity,
                       "row_ids": " ".join(map(str, f.rows)), "related_rows": " ".join(map(str, f.related)),
                       "client": f.position[0], "investing_entity": f.position[1], "fund": f.position[2],
                       "issue": f.issue} for f in findings]).to_csv(out_dir / "findings.csv", index=False)
        g.rename(columns=CHECKS).to_csv(out_dir / "grid.csv")
        print(f"Wrote {out_dir / 'findings.csv'} and {out_dir / 'grid.csv'}")


if __name__ == "__main__":
    main(sys.argv[1:])
