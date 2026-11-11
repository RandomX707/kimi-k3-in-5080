# RTX 5080 port plan — storage-first revision

This fork keeps the upstream CPU engine as the correctness reference and explores whether
an RTX 5080 can materially accelerate Kimi K3 on a consumer workstation.

The revised priority is intentionally **storage -> RAM residency -> measured CPU baseline
-> stop/continue decision -> CUDA**. The reason is simple: at 64 GB, most token traffic
still comes from the slowest tier. A faster kernel cannot recover time spent waiting for
weights that have not arrived.

## Target hardware

- AMD Ryzen 9 5900X (12C/24T)
- NVIDIA GeForce RTX 5080 16 GB
- 64 GB DDR4-3600 today; motherboard maximum 128 GB
- MSI MAG X570 TOMAHAWK WIFI
- Windows 11 Pro today; native Linux is preferred for serious CUDA work
- WD_BLACK SN850X 1 TB NVMe today; insufficient for full K3 deployment
- Corsair RM1200e 1200 W

## Core facts from upstream

K3 is a storage-streaming problem before it is a GPU problem.

Approximate upstream weight traffic at low memory budgets:

- dense trunk: 108.81 GB/token if nothing is pinned;
- routed experts: about 25.83 GB/token;
- total: about 134.64 GB/token.

The released routed experts are already packed MXFP4. The upstream engine keeps the exact
CPU path viable by changing where weights live, not by dropping layers or experts.

Small expert caches are weak because routing is deliberately flat. Therefore, when memory
is scarce, trunk residency is usually more valuable than a small expert LRU.

## Non-negotiable invariants

1. The upstream CPU path remains available as the oracle.
2. A normal upstream build must not require CUDA.
3. No GPU firmware/VBIOS, voltage, permanent power-limit, clock, or driver hacks.
4. Performance claims are end-to-end and replicated at least three times.
5. The full 1.56 TB checkpoint is not downloaded until storage and partial-checkpoint
   gates pass.
6. Every benchmark records SSD temperature/thermal state as well as CPU/GPU/RAM telemetry.
7. We distinguish **bit-exact CPU reference**, **weight/architecture-exact GPU inference**,
   and **speculative/approximate draft paths**. These labels are never conflated.

## What "exact" means

There are three different contracts:

### 1. CPU reference exact

The original upstream CPU path is authoritative. For its deterministic greedy tests,
byte/token identity is the strongest available reference.

### 2. GPU weight/architecture exact

The GPU path uses the same official weights, the same layer graph, the same top-16 routing,
and no post-hoc weight quantisation. Floating-point reduction order and FMA behavior may
differ from CPU, so bit-identical logits are not assumed.

A GPU kernel must pass declared numeric tolerances at operator/layer/logit level and must
show greedy-token agreement on the validation corpus. Any divergence is recorded and
investigated. We do not call numerical closeness "bit exact."

### 3. Verified/speculative exact-output mode

An approximate draft may propose work, but the authoritative path must verify before
emission. This mode is only useful if verification can be amortized enough to retain a
net speedup.

## Operating-system strategy

Upstream now explicitly supports Windows/x86-64 through MSYS2/MinGW and states that
Windows passes its gates. Windows therefore remains a valid baseline and fallback.

For serious CUDA development, however, **native Linux is preferred** because it gives us:

- one GCC/Clang + nvcc toolchain instead of a MinGW host plus MSVC CUDA split;
- simpler direct-I/O and asynchronous I/O behavior;
- simpler pinned-memory/CUDA integration;
- easier profiling with NVIDIA/Linux tooling.

The exact Linux distribution/toolkit version should be chosen only after checking current
NVIDIA support at implementation time.

## Hardware priority

### Priority 1 — storage capacity

The current 1 TB drive cannot hold the 1.56 TB checkpoint plus the packed trunk.

Preferred full-model layout:

- new 4 TB-class high-end NVMe: checkpoint shards;
- existing 1 TB SN850X: packed trunk, source, logs, and test artifacts.

A 2 TB drive is technically possible but leaves little practical headroom.

### Priority 2 — 128 GB RAM before inference CUDA

At 64 GB, the majority of token bytes are still sourced from NVMe. The RTX 5080 can
accelerate arithmetic, but it cannot compute weights that have not arrived.

Moving from 64 GB to 128 GB is therefore the first optional performance upgrade to test
before writing inference kernels.

Important nuance: 128 GB installed does **not** mean unlimited residency. Approximate
memory needs are:

- dense trunk: 108.81 GB;
- fixed model-level weights + recurrent state: about 5.33 GB;
- incremental KV: about 2.37 MB per position;
- plus OS, allocator, staging buffers, and telemetry.

Illustrative exact-path totals before OS/staging:

- 1k context: about 116.6 GB;
- 4k context: about 123.8 GB;
- 8k context: about 133.6 GB.

So 128 GB is a strong short-context trunk-residency tier, not a guarantee of full residency
at long context.

On the current board, reaching 128 GB means replacing the present 4x16 GB configuration
with a validated 4x32 GB configuration. Stability matters more than retaining DDR4-3600.

### Priority 3 — CUDA

CUDA work begins only after the CPU baseline tells us compute is large enough to attack.

## Revised milestone sequence

### M0 — Environment, storage and thermal gate

Run the read-only machine doctor.

Capture:
- CPU, RAM and OS;
- RTX 5080 and driver;
- CUDA compiler/toolkit availability;
- storage capacity/layout;
- NVMe model and temperature/health data where exposed;
- full-checkpoint storage feasibility.

No full checkpoint download.

### M1 — Preserve upstream CPU correctness

Build the untouched CPU engine and run its complete weightless test suite.

Preferred final development OS is native Linux, but Windows baseline evidence remains
useful because it is the user's normal environment.

### M2 — Partial-checkpoint CPU baseline

Use upstream partial-layer download support.

Measure:
- wall time and steady token time where meaningful;
- CPU compute time;
- storage wait time;
- storage bytes/token;
- effective NVMe GB/s;
- CPU utilization and memory bandwidth indicators;
- peak/minimum available RAM;
- pagefile/swap activity;
- SSD temperature and thermal-throttle indicators.

This phase tells us whether our target machine is storage-bound, compute-bound, or mixed.

### M3 — Stop/continue gate

Do **not** automatically proceed to CUDA.

Compute the fraction of end-to-end token time attributable to arithmetic that a GPU could
actually accelerate.

Decision rule:

- if GPU-addressable compute is **<30%** of token time on the 64 GB configuration,
  stop inference-CUDA work on 64 GB and prioritize 128 GB RAM/storage experiments;
- if it is **30-50%**, CUDA is optional and must justify its engineering cost;
- if it is **>50%**, CUDA becomes a strong next lever.

The 30% threshold is a project decision rule, not an upstream fact. It prevents us from
spending weeks optimizing the smallest component.

### M4 — 64 GB vs 128 GB residency experiment

If hardware budget permits, measure the same workload at 64 GB and 128 GB before CUDA.

Questions:
- How much trunk can actually stay resident?
- What is the new storage traffic/token?
- Does expert I/O become the dominant floor?
- What context sizes remain practical without swapping?
- Does CPU-only performance improve enough that CUDA has a larger payoff?

### M5 — CUDA bring-up

Only after M3/M4 says compute is worth attacking.

Build/run the standalone RTX 5080 probe and capture:
- device identity / compute capability;
- runtime and driver;
- usable VRAM;
- pinned H2D/D2H throughput.

### M6 — Backend boundary

Introduce CPU/CUDA dispatch with CPU as the default and oracle.

No model math changes in this milestone.

### M7 — BF16 trunk CUDA

Port the dense-trunk math first.

Acceptance requires:
- operator/layer/logit numeric contract;
- greedy-token validation corpus results;
- end-to-end speedup greater than benchmark noise.

A faster microbenchmark with unchanged end-to-end time is not a success.

### M8 — MXFP4 routed experts

Add Blackwell-aware packed-expert computation without widening the expert bank.

Validate:
- one expert;
- one routed layer;
- partial model;
- full model.

### M9 — I/O / transfer overlap

Pipeline:
- storage read;
- pinned-host staging;
- H2D;
- GPU execution.

Measure actual overlap. Asynchronous APIs alone are not evidence.

### M10 — Speculative R&D

Only after exact GPU inference is stable:
- GPU-resident draft;
- reduced-top-k draft experiments;
- batched verification;
- exact authoritative emission.

## Why 128 GB changes the CUDA economics

At 64 GB, a plausible placement still leaves tens of gigabytes of trunk plus expert
weights coming from NVMe each token. That makes the storage device the dominant clock.

At a short-context 128 GB configuration, most or all trunk bytes may remain resident,
leaving routed-expert traffic as the main storage floor. Only then can arithmetic and
host-to-device movement become a large enough share of token time for the 5080 to matter
substantially.

Even then, PCIe matters. RAM-resident weights that are not also VRAM-resident must cross
PCIe before GPU compute. We therefore measure H2D and compare it with CPU-in-place
execution rather than assuming the GPU is faster.

## SSD thermal policy

Sustained K3 inference is an unusually heavy storage workload.

Every performance run records:
- SSD controller/composite temperature where available;
- effective read throughput over time;
- thermal-throttle or warning counters where available.

The checkpoint NVMe should use an adequate heatsink/airflow path. A benchmark that slows
because the SSD thermally throttles is not a model/kernel result.

## Success labels

### Proof-of-life
The full official K3 text model produces validated output on the workstation.

### Useful systems result
The project establishes a reproducible hardware/software floor and explains which tier
limits it, even if chat remains slow.

### Material acceleration
A new hardware or CUDA configuration produces a repeatable end-to-end improvement larger
than measurement noise.

### Interactive
Normal conversational latency. This is not assumed and is not required for project
success.

## Current implementation status

Already present on branch `feat/rtx5080-cuda-bootstrap`:

- read-only Windows machine doctor;
- standalone CUDA probe;
- CUDA CMake bootstrap;
- initial design documentation.

The upstream CPU inference path is intentionally unchanged.

## Immediate next action

Before more inference code:

1. run the hardware/storage doctor;
2. get the upstream CPU gates green;
3. establish partial-checkpoint CPU telemetry;
4. apply the M3 stop/continue rule;
5. only then choose between RAM/storage work and CUDA.

This is now a **measurement-led systems project**, not a CUDA-first port.
