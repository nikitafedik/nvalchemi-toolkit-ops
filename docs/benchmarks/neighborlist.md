# Neighbor List Benchmarks

Performance benchmarks for neighbor list algorithms in ALCHEMI Toolkit-Ops.
Results show scaling behaviour across system sizes for multiple cutoff radii
and algorithms.

```{warning}
These results are intended to be indicative _only_: your actual performance may
vary depending on the atomic system topology, software and hardware configuration
and we encourage users to benchmark on their own systems of interest.
```

## How to Read These Charts

Time Scaling
: Mean execution time (µs/atom) vs. system size. Lower is better. Cell list
  algorithms show $O(N)$ scaling while naive algorithms show $O(N^2)$.

Throughput
: Atoms processed per second (plotted as 10⁶ atoms/s). Higher is better.
  This metric helps compare efficiency across different system sizes.

Memory
: Peak GPU memory usage vs. system size. Units switch between MB and GB
  automatically on the y-axis. Useful for estimating memory requirements
  for your target system.

## Performance Results

<!-- markdownlint-disable MD013 -->

```{raw} html
<div id="nl-cutoff-switch" style="display:flex; flex-wrap:wrap; gap:0.5rem; align-items:center; margin:0.75rem 0 1rem;">
  <strong>Cutoff view</strong>
  <button type="button" data-cutoff="all" aria-pressed="true">All</button>
  <button type="button" data-cutoff="6A" aria-pressed="false">6 A</button>
  <button type="button" data-cutoff="15A" aria-pressed="false">15 A</button>
  <button type="button" data-cutoff="25A" aria-pressed="false">25 A</button>
  <button type="button" data-cutoff="6A-15A" aria-pressed="false">6 + 15 A</button>
  <button type="button" data-cutoff="6A-25A" aria-pressed="false">6 + 25 A</button>
  <button type="button" data-cutoff="15A-25A" aria-pressed="false">15 + 25 A</button>
</div>
<script>
(function () {
  const root = document.getElementById("nl-cutoff-switch");
  if (!root) {
    return;
  }
  const imagePattern = /^(nl-(?:cscl|nh3)-(?:system-size|constant-workload|batch)-scaling)(?:-cutoff-[^-]+(?:-[^-]+)*)?(-jax)?-(time|throughput|memory)\.png$/;

  function updateImages(cutoff) {
    document.querySelectorAll("img").forEach(function (image) {
      const src = image.getAttribute("src") || "";
      const parts = src.split("/");
      const filename = parts.pop();
      const baseFilename = image.dataset.nlCutoffBase || filename;
      const match = baseFilename.match(imagePattern);
      if (!match) {
        return;
      }
      image.dataset.nlCutoffBase = baseFilename;
      const nextFilename = cutoff === "all"
        ? baseFilename
        : match[1] + "-cutoff-" + cutoff + (match[2] || "") + "-" + match[3] + ".png";
      const baseSrc = parts.concat([baseFilename]).join("/");
      parts.push(nextFilename);
      image.onerror = function () {
        image.onerror = null;
        image.setAttribute("src", baseSrc);
      };
      image.setAttribute("src", parts.join("/"));
      image.setAttribute("alt", (image.getAttribute("alt") || "Neighbor list benchmark plot")
        .replace(/ \(cutoff: [^)]+\)$/, "") + " (cutoff: " + cutoff.replace(/A/g, " A").replace(/-/g, ", ") + ")");
    });
  }

  root.querySelectorAll("button").forEach(function (button) {
    button.addEventListener("click", function () {
      root.querySelectorAll("button").forEach(function (other) {
        other.setAttribute("aria-pressed", "false");
      });
      button.setAttribute("aria-pressed", "true");
      updateImages(button.dataset.cutoff);
    });
  });
})();
</script>
```

<!-- markdownlint-enable MD013 -->

::::{tab-set}

:::{tab-item} Torch
:selected:

`````{tab-set}

````{tab-item} CsCl
:selected:

```{eval-rst}
.. tab-set::

    .. tab-item:: System Size Scaling

        .. figure:: _static/nl-cscl-system-size-scaling-time.png
           :width: 90%
           :align: center

           Mean execution time vs. system size for naive and cell list algorithms.

        .. figure:: _static/nl-cscl-system-size-scaling-throughput.png
           :width: 90%
           :align: center

           Throughput (10⁶ atoms/s) vs. system size.

        .. figure:: _static/nl-cscl-system-size-scaling-memory.png
           :width: 90%
           :align: center

           Peak GPU memory vs. system size.

    .. tab-item:: Constant Workload

        .. figure:: _static/nl-cscl-constant-workload-scaling-time.png
           :width: 90%
           :align: center

           Execution time at constant total atom count, varying batch size.

        .. figure:: _static/nl-cscl-constant-workload-scaling-throughput.png
           :width: 90%
           :align: center

           Throughput at constant total atom count.

        .. figure:: _static/nl-cscl-constant-workload-scaling-memory.png
           :width: 90%
           :align: center

           Peak GPU memory at constant total atom count.

    .. tab-item:: Batch Scaling

        .. figure:: _static/nl-cscl-batch-scaling-time.png
           :width: 90%
           :align: center

           Execution time vs. batch size (fixed atoms per system).

        .. figure:: _static/nl-cscl-batch-scaling-throughput.png
           :width: 90%
           :align: center

           Throughput vs. batch size.

        .. figure:: _static/nl-cscl-batch-scaling-memory.png
           :width: 90%
           :align: center

           Peak GPU memory vs. batch size.
```

````

````{tab-item} NH₃

```{eval-rst}
.. tab-set::

    .. tab-item:: System Size Scaling

        .. figure:: _static/nl-nh3-system-size-scaling-time.png
           :width: 90%
           :align: center

           Mean execution time vs. system size for naive and cell list algorithms (NH₃).

        .. figure:: _static/nl-nh3-system-size-scaling-throughput.png
           :width: 90%
           :align: center

           Throughput (10⁶ atoms/s) vs. system size (NH₃).

        .. figure:: _static/nl-nh3-system-size-scaling-memory.png
           :width: 90%
           :align: center

           Peak GPU memory vs. system size (NH₃).

    .. tab-item:: Constant Workload

        .. figure:: _static/nl-nh3-constant-workload-scaling-time.png
           :width: 90%
           :align: center

           Execution time at constant total atom count (NH₃).

        .. figure:: _static/nl-nh3-constant-workload-scaling-throughput.png
           :width: 90%
           :align: center

           Throughput at constant total atom count (NH₃).

        .. figure:: _static/nl-nh3-constant-workload-scaling-memory.png
           :width: 90%
           :align: center

           Peak GPU memory at constant total atom count (NH₃).

    .. tab-item:: Batch Scaling

        .. figure:: _static/nl-nh3-batch-scaling-time.png
           :width: 90%
           :align: center

           Execution time vs. batch size (NH₃).

        .. figure:: _static/nl-nh3-batch-scaling-throughput.png
           :width: 90%
           :align: center

           Throughput vs. batch size (NH₃).

        .. figure:: _static/nl-nh3-batch-scaling-memory.png
           :width: 90%
           :align: center

           Peak GPU memory vs. batch size (NH₃).
```

````

`````

:::

:::{tab-item} JAX

`````{tab-set}

````{tab-item} CsCl
:selected:

```{eval-rst}
.. tab-set::

    .. tab-item:: System Size Scaling

        .. figure:: _static/nl-cscl-system-size-scaling-jax-time.png
           :width: 90%
           :align: center

           Mean execution time vs. system size (JAX).

        .. figure:: _static/nl-cscl-system-size-scaling-jax-throughput.png
           :width: 90%
           :align: center

           Throughput (10⁶ atoms/s) vs. system size (JAX).

    .. tab-item:: Constant Workload

        .. figure:: _static/nl-cscl-constant-workload-scaling-jax-time.png
           :width: 90%
           :align: center

           Execution time at constant total atom count (JAX).

        .. figure:: _static/nl-cscl-constant-workload-scaling-jax-throughput.png
           :width: 90%
           :align: center

           Throughput at constant total atom count (JAX).

    .. tab-item:: Batch Scaling

        .. figure:: _static/nl-cscl-batch-scaling-jax-time.png
           :width: 90%
           :align: center

           Execution time vs. batch size (JAX).

        .. figure:: _static/nl-cscl-batch-scaling-jax-throughput.png
           :width: 90%
           :align: center

           Throughput vs. batch size (JAX).

```

```{note}
JAX memory plots are omitted. The suite does not measure JAX memory:
XLA's allocator — whether the default BFC pool (which pre-allocates most
of VRAM) or the on-demand variant (which fragments on retry) — makes
per-call memory attribution unreliable. Torch memory plots are
representative; both backends dispatch through identical Warp GPU
kernels with the same memory footprint.
```

````

````{tab-item} NH₃

```{eval-rst}
.. tab-set::

    .. tab-item:: System Size Scaling

        .. figure:: _static/nl-nh3-system-size-scaling-jax-time.png
           :width: 90%
           :align: center

           Mean execution time vs. system size (JAX, NH₃).

        .. figure:: _static/nl-nh3-system-size-scaling-jax-throughput.png
           :width: 90%
           :align: center

           Throughput (10⁶ atoms/s) vs. system size (JAX, NH₃).

    .. tab-item:: Constant Workload

        .. figure:: _static/nl-nh3-constant-workload-scaling-jax-time.png
           :width: 90%
           :align: center

           Execution time at constant total atom count (JAX, NH₃).

        .. figure:: _static/nl-nh3-constant-workload-scaling-jax-throughput.png
           :width: 90%
           :align: center

           Throughput at constant total atom count (JAX, NH₃).

    .. tab-item:: Batch Scaling

        .. figure:: _static/nl-nh3-batch-scaling-jax-time.png
           :width: 90%
           :align: center

           Execution time vs. batch size (JAX, NH₃).

        .. figure:: _static/nl-nh3-batch-scaling-jax-throughput.png
           :width: 90%
           :align: center

           Throughput vs. batch size (JAX, NH₃).

```

```{note}
JAX memory plots are omitted. The suite does not measure JAX memory:
XLA's allocator — whether the default BFC pool (which pre-allocates most
of VRAM) or the on-demand variant (which fragments on retry) — makes
per-call memory attribution unreliable. Torch memory plots are
representative; both backends dispatch through identical Warp GPU
kernels with the same memory footprint.
```

````

`````

:::

:::{tab-item} Backend Comparison

The reportable NL suite does not emit Torch-vs-JAX overlay plots. The JAX
reportable rows use serial `block_until_ready()` timing for large configurations
so the suite does not retain ten large JAX outputs at once; those timings are
valid JAX measurements, but they are not a like-for-like overlay against Torch
CUDA-event timings. Use the Torch and JAX tabs above for backend-specific
scaling, throughput, and cutoff views from the same CSVs.

:::

::::

## Method Variants

The NL suite treats each user-facing neighbor-list strategy as a separate
method. The naive family contains `naive_scalar` and `naive_tile`; the cell-list
family contains `cell_list_atom_centric` and `cell_list_pair_centric`.
`cluster_tile` is the Morton-sorted tile algorithm added in the 0.4 line.
Batched inputs use the matching batch API where needed, but the CSV method
column keeps the concrete strategy visible for plotting and comparison.

## Benchmark Configuration

| Parameter | Value |
| --------- | ----- |
| Cutoffs | 6.0, 15.0, 25.0 Å |
| Methods | `naive_scalar`, `naive_tile`, `cell_list_atom_centric`, `cell_list_pair_centric`, `cluster_tile` |
| System Type | CsCl (programmatic), NH₃ (PDB) |
| Warmup Iterations | 3 |
| Timing Iterations | 10 |
| Dtype | `float32` |

## Running Your Own Benchmarks

Run from the repository root:

```bash
python -m benchmarks.neighborlist.benchmark_neighborlist \
    --config benchmarks/neighborlist/benchmark_config.yaml \
    --output-dir docs/benchmarks/benchmark_results
```

For the JAX backend:

```bash
python -m benchmarks.neighborlist.benchmark_neighborlist \
    --config benchmarks/neighborlist/benchmark_config.yaml \
    --backend jax \
    --output-dir docs/benchmarks/benchmark_results
```

For direct Warp API timing:

```bash
python -m benchmarks.neighborlist.benchmark_neighborlist \
    --config benchmarks/neighborlist/benchmark_config.yaml \
    --backend warp \
    --method cell_list_atom_centric \
    --output-dir docs/benchmarks/benchmark_results
```

Use `--method` / `--methods` to restrict the benchmark to particular APIs, for
example `--method naive_tile cell_list_pair_centric`, and `--dry-run` to inspect
the expanded `(system, mode, method, cutoff)` plan before allocating GPU memory.
