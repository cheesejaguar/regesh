"""Terminal installer: a curses checklist of detected harnesses, with a plain-text fallback."""
from __future__ import annotations
import sys

BANNER = "regesh  ·  turn any agent into an emotional agent"
HELP = "↑/↓ or j/k move   space toggle   a all   enter install   q quit"


def format_report(report):
    lines = [f"{report['title']}:"]
    for change in report["changes"]:
        lines.append(f"  {change['action']:<20} {change['file']}")
    for note in report["notes"]:
        lines.append(f"  note: {note}")
    return lines


def plain_install(installer, harnesses, selected, out=sys.stdout):
    reports = []
    for h in harnesses:
        if h["id"] in selected:
            reports.append(installer.install(h["id"]))
    for r in reports:
        out.write("\n".join(format_report(r)) + "\n")
    return reports


def run_curses(installer, harnesses, preselected):
    import curses

    def main(stdscr):
        curses.curs_set(0)
        stdscr.keypad(True)
        selected = set(preselected)
        cursor = 0
        reports = None
        while True:
            stdscr.erase()
            h, w = stdscr.getmaxyx()
            stdscr.addnstr(0, 0, BANNER, w - 1, curses.A_BOLD)
            stdscr.addnstr(1, 0, "Detected agent harnesses. Selected ones get the regesh MCP server, hooks and an instruction block.", w - 1)
            row = 3
            if reports is None:
                for i, item in enumerate(harnesses):
                    mark = "x" if item["id"] in selected else " "
                    status = ("detected: " + item["evidence"][0]) if item["detected"] else "not detected (select to configure anyway)"
                    attr = curses.A_REVERSE if i == cursor else (curses.A_NORMAL if item["detected"] else curses.A_DIM)
                    stdscr.addnstr(row, 0, f" [{mark}] {item['title']:<14} {status}", w - 1, attr); row += 1
                    if i == cursor:
                        for s in item["surfaces"]:
                            stdscr.addnstr(row, 6, f"· {s}", w - 7, curses.A_DIM); row += 1
                stdscr.addnstr(row + 1, 0, f"State lives in {installer.home}; every edited file is backed up; undo with: regesh-install --uninstall", w - 1, curses.A_DIM)
                stdscr.addnstr(h - 1, 0, HELP, w - 1, curses.A_DIM)
            else:
                for r in reports:
                    for line in format_report(r):
                        if row < h - 2:
                            stdscr.addnstr(row, 0, line, w - 1); row += 1
                    row += 1
                stdscr.addnstr(h - 1, 0, "Done. Press any key to exit.", w - 1, curses.A_BOLD)
            stdscr.refresh()
            key = stdscr.getch()
            if reports is not None:
                return reports
            if key in (ord("q"), 27):
                return []
            if key in (curses.KEY_UP, ord("k")):
                cursor = (cursor - 1) % len(harnesses)
            elif key in (curses.KEY_DOWN, ord("j")):
                cursor = (cursor + 1) % len(harnesses)
            elif key == ord(" "):
                hid = harnesses[cursor]["id"]
                selected.symmetric_difference_update({hid})
            elif key == ord("a"):
                selected = set(x["id"] for x in harnesses) if len(selected) < len(harnesses) else set()
            elif key in (curses.KEY_ENTER, 10, 13):
                if not selected:
                    continue
                stdscr.addnstr(h - 1, 0, "Installing…", w - 1, curses.A_BOLD); stdscr.refresh()
                reports = [installer.install(x["id"]) for x in harnesses if x["id"] in selected]
    return curses.wrapper(main)
