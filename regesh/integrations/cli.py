"""``regesh-install``: detect installed agent harnesses and hook the affect layer into them."""
from __future__ import annotations
import argparse
import json
import os
import sys

from .detect import HARNESSES, ORDER, detect
from .install import Installer
from . import tui


def parse_harnesses(values):
    out = []
    for v in values or []:
        for part in v.split(","):
            part = part.strip().lower()
            if not part:
                continue
            if part not in HARNESSES:
                raise SystemExit(f"unknown harness {part!r}; choose from {', '.join(ORDER)}")
            if part not in out:
                out.append(part)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(prog="regesh-install", description="Install the regesh affect layer into agent harnesses.")
    parser.add_argument("--harness", action="append", help="harness id(s), comma separated: " + ", ".join(ORDER))
    parser.add_argument("--all", action="store_true", help="configure every supported harness, detected or not")
    parser.add_argument("-y", "--yes", action="store_true", help="no interface: install into every detected harness")
    parser.add_argument("--uninstall", action="store_true", help="remove the regesh hooks, servers and instruction blocks")
    parser.add_argument("--dry-run", action="store_true", help="show what would change without writing")
    parser.add_argument("--list", action="store_true", help="show detected and installed harnesses")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--home", default=None, help="regesh state directory (default ~/.regesh or $REGESH_HOME)")
    parser.add_argument("--user-home", default=None, help="home directory holding the harness configs (default ~)")
    parser.add_argument("--bin-dir", default=None, help="directory holding regesh-mcp/regesh-hook (default: this interpreter's)")
    args = parser.parse_args(argv)

    installer = Installer(home=args.home, user_home=args.user_home, bin_dir=args.bin_dir, dry_run=args.dry_run)
    harnesses = detect(home=installer.user_home)
    installed = set(installer.installed())
    for h in harnesses:
        h["installed"] = h["id"] in installed

    if args.list:
        if args.json:
            print(json.dumps(harnesses, indent=1)); return 0
        for h in harnesses:
            flag = "detected" if h["detected"] else "absent  "
            print(f"{h['id']:<9} {h['title']:<14} {flag} {'installed' if h['installed'] else ''}  {'; '.join(h['evidence'])}")
        return 0

    chosen = parse_harnesses(args.harness)
    if args.all:
        chosen = list(ORDER)

    if args.uninstall:
        targets = chosen or sorted(installed, key=ORDER.index)
        results = [installer.uninstall(h) for h in targets]
        if args.json:
            print(json.dumps(results, indent=1))
        else:
            for r in results:
                print(f"removed regesh from {r['title']}" + (f" (backups: {', '.join(r['backups'])})" if r["backups"] else ""))
            if not results:
                print("nothing to uninstall")
        return 0

    detected = [h["id"] for h in harnesses if h["detected"]]
    interactive = sys.stdin.isatty() and sys.stdout.isatty() and not args.yes and not chosen and not args.all and not args.json and os.environ.get("TERM", "dumb") != "dumb"
    if interactive:
        try:
            reports = tui.run_curses(installer, harnesses, detected)
        except Exception as exc:  # curses unavailable or terminal too small: fall back to plain output
            print(f"(terminal interface unavailable: {exc}; installing into detected harnesses)")
            reports = tui.plain_install(installer, harnesses, set(detected))
        if not reports:
            print("nothing installed"); return 0
        print("\n".join(line for r in reports for line in tui.format_report(r)))
        return 0

    selection = set(chosen) if chosen else set(detected)
    if not selection:
        print("no supported agent harness detected; pass --harness <id> (" + ", ".join(ORDER) + ") or --all"); return 1
    reports = [installer.install(h) for h in ORDER if h in selection]
    if args.json:
        print(json.dumps(reports, indent=1))
    else:
        for r in reports:
            print("\n".join(tui.format_report(r)))
        print(f"\nstate: {installer.home}   undo: regesh-install --uninstall")
    return 0


if __name__ == "__main__":
    sys.exit(main())
