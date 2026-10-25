# Compact memory journal and optional Rust kernel

This change targets an observed storage bottleneck, not a new affect mechanism.
The original journal wrote the complete growing state after every operation,
including the entire observation/memory history. Merely translating that code to
Rust would preserve quadratic journal growth.

## What changed

`Store.append` now writes a full initial checkpoint followed by lossless nested
state deltas, using a full checkpoint only when it is smaller than its delta.
Dictionary additions/deletions, replacements and append-only list suffixes are
represented explicitly. The materialized `agents.state` row is still one complete
current state. SQLite's existing transaction encloses both the journal and current
state update. Public state schema v1, snapshots, event identity, outcomes and every
logical journal hash remain unchanged. Legacy full journals and mixed legacy /
compact histories are readable without rewriting old evidence.

The Rust crate `native/` implements the JSON-difference kernel through a small
`ctypes` C ABI. It is a real integrated backend, used by each journal write when
selected. Python still supplies the SDK, appraisal, SQLite transactions and
reconstruction. This is not a complete Rust port. A byte-for-byte-equivalent
Python delta backend is included, so compact storage never requires a compiler.
Both kernels preserve integer-versus-float distinctions, signed zero, Unicode,
append behavior, and arbitrary-sized JSON integers. Rust uses `float_roundtrip`
and `arbitrary_precision`. Deep Python-valid evidence beyond serde's parser depth and escaped lone surrogates accepted by Python use an observable Python fallback (`DeltaKernel.native_fallbacks`). Invalid
native results otherwise raise rather than silently changing state.

`Store.iter_entries` streams reconstructed independent poststates. `entries()`
retains its compatibility list interface. `Agent.snapshot` now obtains only the
latest hash, avoiding its previous accidental construction of the entire full
journal. Taking a snapshot is not a full historical integrity audit; use streaming
`iter_entries` to audit all links. Default replay now streams logical entries through `Agent.iter_trace`; `trace()` and the inspector still explicitly request complete logical traces and can retain substantial memory.

## Build and use

```bash
cargo test --manifest-path native/Cargo.toml
cargo build --release --manifest-path native/Cargo.toml
python -m unittest discover -s tests -p test_native_memory.py -v
REGESH_MEMORY_BACKEND=rust python -m regesh demo --db native-demo.sqlite
```

`REGESH_MEMORY_BACKEND` accepts `python`, `rust`, `auto` and `snapshot`. The
default is Python compact storage because Rust did not show a dependable
latency/RSS advantage. Explicit `auto`
uses a built library in the checkout when available, otherwise Python compact
storage. Explicit `rust` fails clearly if the library is absent. `snapshot` is
the full-journal comparison backend, not a new cognition condition. For installed
wheels or a custom build path, set `REGESH_NATIVE_LIBRARY` to the compiled shared
library. Compiler build products are ignored; source and Cargo.lock are versioned.
On the measured Linux CPU, the optimized shared library is about 610 KiB.

## Reproducible profiling protocol

```bash
python scripts/profile_memory.py --output results/memory-profile-new \
  --counts 25 100 250 --repeats 2
```

The profile compares full snapshots, Python compact deltas and Rust compact
deltas with identical evidence and exact final-state/journal-hash checks. Each
measurement runs in a new subprocess. Backend order rotates between repetitions.
The two workloads are sparse append-only observations and the common clocked
history `observe → advance_time(1)`. Metrics are write latency, snapshot latency,
streaming audit latency, checkpointed SQLite bytes and process peak RSS. Database
bytes exclude the truncated WAL; RSS includes the interpreter and native parser,
not just memory payloads. Timings are descriptive medians of two repetitions,
not confidence intervals or broad hardware performance claims.

The official model-inference workload was held until profiling completed. The
background llama.cpp compilation was paused for these measurements. Source and
runtime metadata accompany the result. The original full-snapshot backend also
benefits from common hash-serialization and snapshot-lookup improvements; this
comparison isolates representation/kernel differences rather than attributing
all old-versus-new PR differences to Rust.

## Limits that matter

The strongest expected saving is **disk retention from delta representation**.
Python compact storage gets the same representation saving. Rust can change CPU
cost but does not automatically reduce resident memory: the Python state and a
native parsed tree coexist during the FFI call. Choose Python compact storage
when its measured memory/latency tradeoff is preferable.

Sparse appended histories have approximately linear retained journal content.
That is not a universal complexity claim. `advance_time` changes every historical
affect record and habituation value, so clocked trajectories can still have
quadratic changed-leaf volume. Lossless deltas reduce the constant but do not
eliminate that problem. Python still materializes and validates the whole current
state on each operation. A future stronger design would store immutable episodes
separately, keep a bounded working set, and lazily evaluate decays from timestamps;
that requires explicit compatibility rules and is not silently implemented here.

No memory is forgotten and no evidence is silently dropped. These changes do not
improve psychological validity, affect quality or creativity by themselves.

## Executed measurements

The retained evidence is `results/memory-profile/` (36 measurements before the
redundant native-input Python parse was removed), `results/memory-final-focus/`
(eight final-kernel measurements at 100 events), and `results/snapshot-profile/`
(one fresh process per original/new snapshot lookup). Source snapshots, hashes,
raw rows and runtime context accompany them. An exceptional parser-depth fallback
counter was added during the first cohort; no input entered that error path. The
first cohort is explicitly pre-optimization; final-code timing is the focused
cohort. Changing the default backend does not affect explicitly selected backends.

| Workload, 250 events | Full journal bytes | Python/Rust compact bytes | Retention reduction |
|---|---:|---:|---:|
| Append observations | 94,457,856 | 2,727,936 | 97.1% |
| Observation + elapsed time | 187,699,200 | 6,406,144 | 96.6% |

Every backend had the exact same logical state and journal hash. The disk saving
comes from representation, equally available in Python. It is **not a Rust-specific
saving**. At 25/100/250 events compact clocked retention was 352,256 / 1,765,376 /
6,406,144 bytes: dense time evolution still increases retained changed-leaf cost
faster than the event count. No universal linear scaling claim is supported.

In the final focused 100-event run, median append write time was **1.768s Python
vs 3.083s Rust**; clocked write time was **3.384s vs 3.285s**. Rust's peak process
RSS was higher (about 23–24MiB versus Python's 20–21MiB). The two-repetition wall
timings were noisy. **The Rust prototype did not earn default selection**: these
measurements establish neither a general CPU speedup nor a RAM advantage. Native
parsing and FFI trees can erase lower-level computation gains. It remains an
optional fully tested kernel for further profiling, not a performance claim.

A separate 250-event comparison reproduces the original snapshot lookup
`entries()[-1]` versus the new latest-hash lookup. Peak RSS was **235,460KiB vs
24,608KiB**, an **89.5%** reduction in this isolated measurement; measured snapshot
latency was **7.372s vs 0.108s**. Results and journal hash were identical. This is
one measurement each, not a confidence interval, and the improvement is an
algorithmic removal of full-history materialization rather than a language effect.
The materialized current state still grows with preserved evidence.

Two Rust unit tests and eleven Python native/storage tests passed, including 300
random nested JSON comparisons, signed zero, integer/float distinctions,
arbitrary-sized integers, parser-depth fallback, legacy/compact mixed histories,
exact hashes, corruption rejection, replay, independent streamed states and the
snapshot materialization regression. The complete project suite also passed
at this revision. No cognitive or creativity result is inferred from storage
performance.
