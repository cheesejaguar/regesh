"""Figures for the pair-bonding extension from saved summaries only (no simulation)."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def platform_panels(summary, axes):
    att = summary["attachment"]
    qs = sorted({float(k.split("|")[0][2:]) for k in att["conditions"]}, reverse=True)
    ax = axes[0]
    for key, color in (("final_responsiveness", "#1f77b4"), ("final_security", "#ff7f0e"), ("mean_threat_buffer", "#2ca02c")):
        means = [att["conditions"][f"q={q}|bonded"][key]["mean"] for q in qs]
        lo = [att["conditions"][f"q={q}|bonded"][key]["ci95"][0] for q in qs]
        hi = [att["conditions"][f"q={q}|bonded"][key]["ci95"][1] for q in qs]
        ax.errorbar(qs, means, yerr=[[m - l for m, l in zip(means, lo)], [h - m for m, h in zip(means, hi)]], marker="o", label=key.replace("_", " "), color=color, capsize=3)
    ax.set_xlabel("partner responsiveness (probability of answering a request)"); ax.set_ylabel("value at end of run"); ax.set_title("A. Attachment: measured responsiveness, security, buffer"); ax.legend(fontsize=8)
    ax = axes[1]
    diffs = [att["paired_bonded_minus_reference"][f"q={q}"]["cumulative_threat"] for q in qs]
    ax.bar([str(q) for q in qs], [d["mean"] for d in diffs], yerr=[[d["mean"] - d["ci95"][0] for d in diffs], [d["ci95"][1] - d["mean"] for d in diffs]], color="#9467bd", capsize=4)
    ax.axhline(0, color="k", lw=.8); ax.set_xlabel("partner responsiveness"); ax.set_ylabel("cumulative affect threat, bonded minus reference"); ax.set_title("B. Secure-base buffering (paired, common random numbers)")
    ax = axes[2]
    dy = summary["dyad"]["conditions"]
    partners = sorted({k.split("|")[0] for k in dy}); policies = ["own_mood", "mirror", "steer", "steer_tight", "steer_no_repair"]
    width = .8 / len(policies)
    for i, pol in enumerate(policies):
        vals = [dy[f"{p}|{pol}"]["final_partner_affect_mean"] for p in partners]
        ax.bar([j + i * width for j in range(len(partners))], [v["mean"] for v in vals],
               yerr=[[v["mean"] - v["ci95"][0] for v in vals], [v["ci95"][1] - v["mean"] for v in vals]], width=width, label=pol, capsize=2)
    ax.set_xticks([j + .4 - width / 2 for j in range(len(partners))]); ax.set_xticklabels([p.replace("_", "\n") for p in partners], fontsize=8)
    ax.axhline(0, color="k", lw=.8); ax.set_ylabel("partner affect, mean of final 40 turns"); ax.set_title("C. Dyadic steering versus mirroring"); ax.legend(fontsize=7)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", default="results/bonding/platform/summary.json")
    parser.add_argument("--output", default="results/bonding/figures")
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    summary = json.loads(Path(args.platform).read_text())
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    platform_panels(summary, axes)
    fig.suptitle("Pair-bonding platform checks in simulation (no human partner; see docs/bonding.md)", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out / f"platform.{ext}", dpi=140)
    manifest = {"source": args.platform, "files": sorted(p.name for p in out.iterdir())}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
