"""Figures for the pair-bonding extension from saved summaries only (no simulation)."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def platform_panels(summary, axes, label=""):
    att = summary["attachment"]
    qs = sorted({float(k.split("|")[0][2:]) for k in att["conditions"]}, reverse=True)
    variants = sorted({k.split("|")[1] for k in att["conditions"] if k.split("|")[1] != "reference"})
    ax = axes[0]
    styles = ["-", "--", ":"]
    for vi, variant in enumerate(variants):
        for key, color in (("final_responsiveness", "#1f77b4"), ("mean_threat_buffer", "#2ca02c")):
            means = [att["conditions"][f"q={q}|{variant}"][key]["mean"] for q in qs]
            lo = [att["conditions"][f"q={q}|{variant}"][key]["ci95"][0] for q in qs]
            hi = [att["conditions"][f"q={q}|{variant}"][key]["ci95"][1] for q in qs]
            ax.errorbar(qs, means, yerr=[[m - l for m, l in zip(means, lo)], [h - m for m, h in zip(means, hi)]], marker="o", ls=styles[vi % 3],
                        label=f"{key.replace('_', ' ')} ({variant.replace('bonded:', '') or 'bonded'})", color=color, capsize=3)
    ax.set_xlabel("partner responsiveness (probability of answering a request)"); ax.set_ylabel("value at end of run"); ax.set_title(f"A{label}. Attachment: measured responsiveness and buffer"); ax.legend(fontsize=7)
    ax = axes[1]
    pairs = att["paired_bonded_minus_reference"]
    keys = sorted(pairs, key=lambda k: (-float(k.split("|")[0][2:]), k))
    diffs = [pairs[k]["cumulative_threat"] for k in keys]
    ax.bar([k.replace("q=", "q ").replace("|", "\n") for k in keys], [d["mean"] for d in diffs], yerr=[[d["mean"] - d["ci95"][0] for d in diffs], [d["ci95"][1] - d["mean"] for d in diffs]], color="#9467bd", capsize=4)
    ax.tick_params(axis="x", labelsize=7)
    ax.axhline(0, color="k", lw=.8); ax.set_xlabel("partner responsiveness | variant"); ax.set_ylabel("cumulative affect threat, bonded minus reference"); ax.set_title(f"B{label}. Secure-base buffering (paired)")
    ax = axes[2]
    dy = summary["dyad"]["conditions"]
    partners = sorted({k.split("|")[0] for k in dy}); policies = [p for p in ["own_mood", "mirror", "fixed_positive", "steer", "steer_tight", "steer_no_repair"] if any(k.endswith("|" + p) for k in dy)]
    width = .8 / len(policies)
    for i, pol in enumerate(policies):
        vals = [dy[f"{p}|{pol}"]["final_partner_affect_mean"] for p in partners]
        ax.bar([j + i * width for j in range(len(partners))], [v["mean"] for v in vals],
               yerr=[[v["mean"] - v["ci95"][0] for v in vals], [v["ci95"][1] - v["mean"] for v in vals]], width=width, label=pol, capsize=2)
    ax.set_xticks([j + .4 - width / 2 for j in range(len(partners))]); ax.set_xticklabels([p.replace("_", "\n") for p in partners], fontsize=7)
    ax.axhline(0, color="k", lw=.8); ax.set_ylabel("partner affect, mean of final 40 turns"); ax.set_title(f"C{label}. Dyadic steering versus mirroring and fixed positivity"); ax.legend(fontsize=6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", default="results/bonding/platform/summary.json")
    parser.add_argument("--platform-v2", default="results/bonding/platform_v2/summary.json")
    parser.add_argument("--output", default="results/bonding/figures")
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    sources = [(Path(args.platform), "1", "platform"), (Path(args.platform_v2), "2", "platform_v2")]
    for path, label, name in sources:
        if not path.exists():
            continue
        summary = json.loads(path.read_text())
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
        platform_panels(summary, axes, label)
        fig.suptitle(f"Pair-bonding platform checks, protocol {label}, in simulation (no human partner; see docs/bonding.md)", fontsize=10)
        fig.tight_layout()
        for ext in ("png", "svg"):
            fig.savefig(out / f"{name}.{ext}", dpi=140)
    manifest = {"sources": [str(p) for p, _, _ in sources if p.exists()], "files": sorted(p.name for p in out.iterdir() if p.name != "manifest.json")}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
