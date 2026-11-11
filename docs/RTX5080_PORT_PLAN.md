# RTX 5080 port plan

This fork keeps the upstream CPU engine as the correctness reference and adds RTX 5080
acceleration incrementally. The target machine for the first implementation is:

- AMD Ryzen 9 5900X (12C/24T)
- NVIDIA GeForce RTX 5080 16 GB
- 64 GB DDR4-3600
- MSI MAG X570 TOMAHAWK WIFI (PCIe 4.0 x16 GPU link)
- Windows 11 Pro
- WD_BLACK SN850X 1 TB NVMe (current storage; insufficient for the full K3 checkpoint)
- Corsair RM1200e 1200 W

## Non-negotiable invariants

1. The existing CPU path remains available and is the oracle.
2. A normal upstream build must not require CUDA.
3. CUDA work is opt-in and must not change GPU firmware, VBIOS, voltage, power limits,
   driver settings, or any permanent GPU configuration.
4. Exact-mode GPU output must be checked against the CPU reference before a speedup is
   accepted.
5. Benchmark claims require at least three repeated runs. Upstream has measured large
   run-to-run variance, so one fast sample is not evidence.
6. Do not download the complete 1.56 TB checkpoint until storage and partial-checkpoint
   gates pass.

## Why this is a hybrid storage problem

K3 is not a conventional "fit the quant in VRAM" model. The released checkpoint contains
roughly 1.45 TB of routed experts and about 108.81 GB of dense trunk weights. The upstream
engine succeeds on small-memory machines by streaming weights from storage.

The RTX 5080 therefore helps in three different ways:

- compute BF16 trunk operations faster;
- compute native packed expert operations faster;
- use part of 16 GB VRAM as a high-value resident cache.

It cannot remove the need to move tens of gigabytes of weights per generated token.

## Windows toolchain split

Upstream Windows builds use MinGW/MSYS2. NVIDIA's supported Windows CUDA development path
uses MSVC/Visual Studio. Do not force nvcc into the upstream MinGW build.

For the bootstrap phase the CUDA probe is a separate CMake project under `cuda/`. This
lets us verify the RTX 5080 toolchain independently while `make test` continues to use
the unmodified upstream path.

Once the probe and CPU baseline are measured, the integration layer can expose a small C
ABI from an MSVC-built CUDA DLL to the MinGW C engine, or we can port the host build to a
second supported MSVC configuration. We will choose based on measured complexity and
correctness, not preference.

## Milestones

### M0 - Environment and storage gate

Run `scripts/rtx5080-doctor.ps1`.

Required evidence:
- RTX 5080 detected;
- NVIDIA driver reported;
- CUDA compiler/toolkit presence reported;
- Visual Studio/MSVC presence reported;
- physical RAM reported;
- storage layout reported;
- full-checkpoint storage gate reported.

Current known blocker: the listed 1 TB SN850X cannot hold the 1.56 TB checkpoint plus the
packed trunk. A second NVMe (2 TB minimum; 4 TB preferred for practical headroom) is
required before full-model deployment.

### M1 - CUDA bring-up

Build and run `cuda/rtx5080_probe.cu`.

Capture:
- GPU name;
- compute capability;
- runtime and driver versions;
- total/free VRAM;
- measured pinned-host H2D and D2H bandwidth.

The RTX 5080 should report compute capability 12.0. A mismatch is a stop-and-investigate
condition, not something to ignore.

### M2 - Preserve upstream correctness

On the target PC:
- build upstream unchanged;
- run the full weightless test suite;
- save output and compiler versions.

No CUDA inference changes until this gate is green.

### M3 - Partial-checkpoint baseline

Use upstream's partial-layer download support rather than the 1.56 TB full checkpoint.
Measure CPU-only:
- cold start;
- trunk bytes read;
- expert bytes read;
- wall time;
- peak RSS;
- SSD throughput;
- output/reference agreement.

This establishes whether Windows I/O behavior matches upstream assumptions.

### M4 - Backend boundary

Introduce a backend interface with CPU as the default:
- CPU implementation remains byte-for-byte behaviorally unchanged;
- CUDA backend is selected only by an explicit flag;
- allocation, stream, and error handling are isolated from model logic.

### M5 - BF16 trunk acceleration

First real inference kernel target:
- BF16 trunk matrix-vector/matrix operations;
- pinned host staging;
- asynchronous copies;
- two or more CUDA streams;
- no expert CUDA path yet.

Acceptance:
- reference logits/tokens within the exact-mode contract;
- measured end-to-end improvement, not merely a faster microbenchmark.

### M6 - VRAM placement

Use VRAM for trunk residency before a small expert LRU.

Reason: upstream measurements show that small expert caches have weak retention while
trunk bytes are revisited every token. Placement must be evidence-driven and exposed in
telemetry.

### M7 - MXFP4 routed experts

Add a Blackwell-aware expert path while keeping experts packed. Never expand each expert
to FP32/BF16 in host memory.

Tasks:
- verify exact released MXFP4 nibble/scale layout;
- add a layout adapter/repack step only if necessary;
- validate one expert against the CPU kernel;
- validate one real layer;
- validate full-model tokens.

### M8 - Overlapped NVMe -> RAM -> GPU pipeline

Use double/triple buffering:
- storage reads next payload;
- pinned RAM holds staged payload;
- PCIe transfer overlaps GPU compute;
- GPU consumes the current payload.

Measure queue depth and overlap. Do not assume asynchronous APIs mean useful overlap.

### M9 - Speculative research path

Only after exact CUDA inference is stable:
- GPU-resident INT8 draft trunk;
- reduced-top-k draft experiments;
- exact BF16/MXFP4 verification remains authoritative.

Final emitted tokens must remain identical to exact greedy output for any mode advertised
as exact.

## Success criteria

The project is successful even if the final throughput is still below interactive-chat
speed, provided it demonstrates a reproducible improvement over the upstream CPU path and
documents the limiting resource correctly.

We will distinguish:

- **proof-of-life**: full K3 produces correct tokens;
- **usable experiment**: repeatable speedup with bounded resource use;
- **interactive**: latency low enough for normal chat.

Do not collapse those three labels into one.
