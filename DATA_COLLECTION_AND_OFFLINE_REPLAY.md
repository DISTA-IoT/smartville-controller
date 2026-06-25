# Data Collection Mode & Offline Replay/Training

This document describes two related pieces of functionality built around
`TigerBrain` (`tiger_brain_new.py`):

1. **Data collection mode** — record every fully-labelled flow batch the live
   controller sees, to disk, instead of (or alongside) training on it online.
2. **Offline replay** (`offline_replay.py`) — re-feed those recorded batches
   into a freshly-constructed `TigerBrain`, outside the live controller
   process, through the exact same training/inference code path the online
   run used.

The motivating idea: capture real, fully-labelled traffic once, then iterate
on DRL/training changes against that fixed dataset, repeatably and offline,
without needing a live GNS3 topology running.

---

## 1. Data collection mode

### Enabling it

Driven entirely by Hydra config under `intrusion_detection` (see
`tiger/config/overrides/data_collection.yaml` for a ready-made profile):

| Key | Default | Meaning |
|---|---|---|
| `data_collection_mode` | `false` | Master switch. When `true`, `process_input` records every batch instead of training on it. |
| `data_collection_dir` | `/tmp/tiger_data_collection/` | Root directory; a new `run_<timestamp>/` subdirectory is created per run. |
| `data_collection_shard_size` | `2000` | Samples buffered before a shard is flushed to disk. |
| `data_collection_flush_interval_secs` | `30` | Time-based flush trigger, in case traffic is too sparse to fill a shard quickly. |
| `data_collection_use_packet_feats` | `false` | Also persist `packet_features` (only if the model config (`use_packet_feats`) also wants them). |
| `data_collection_use_node_feats` | `false` | Also persist `node_features` (only if the model config (`node_features`) also wants them). |
| `data_collection_skip_training` | `true` | If `true`, `process_input` returns immediately after recording — no replay-buffer push, no inference, no training. If `false`, the batch is recorded **and** still pushed through the normal training path. |

The `data_collection` override profile also disables wandb tracking
(`wandb.wb_tracking: false`) and sets `intrusion_detection.agency: false`,
`pretrained_inference: false`, since a pure capture run doesn't need any of
that.

### What gets recorded

`TigerBrain.__init__` constructs a `FlowDataRecorder` (`data_recorder.py`) at
startup if `data_collection_mode` is true. `process_input` then branches:

```python
if self.data_collection_mode and self.data_recorder is not None:
    batch = self.stack_flow_tensors(flows, node_feats)   # tensors only, no encoder side effects
    tick = self.wb_tracker.step_counter
    self._record_flows_for_data_collection(batch, flows, tick)
    if self.data_collection_skip_training:
        return
    batch.class_labels = self.get_labels(flows)
    ...
```

`stack_flow_tensors` deliberately does **not** touch the dynamic label
encoder or knowledge base — recording must not have side effects on
`current_known_classes_count`/replay buffers. `_record_flows_for_data_collection`
hands the recorder:

- `flow_features` (float32), `packet_features` (uint8, if enabled),
  `node_features` (float32, if enabled) — exactly the tensors that would
  have been used for training.
- `element_classes` — the raw ground-truth NL string label per sample
  (`flow.element_class`).
- `flow_ids` — for traceability.
- `tick` — `wb_tracker.step_counter` at the moment the batch was observed;
  preserves arrival order.

**Important:** `zda`/`test_zda` labels are deliberately **never** recorded.
Those booleans are derived from the run's curriculum state
(`current_knowledge['G1s'/'G2s']`), which mutates whenever that run's agent
buys CTI (`NewTigerEnvironment.perform_epistemic_action`). Baking them into
the dataset would permanently couple any future replay to one specific
agent's purchase history. They are recomputed at replay time from whatever
curriculum the replaying run is configured with — exactly what
`TigerBrain.get_zda_labels` does online.

### On-disk layout

```
<data_collection_dir>/run_<timestamp>/
    manifest.json        # written once, full config snapshot (see below)
    shards_index.jsonl   # one JSON line appended per shard flushed
    shard_000000.pt       # torch.save'd dict
    shard_000001.pt
    ...
```

Each shard (`torch.load`-able dict) contains:

```python
{
    "flow_features": Tensor[N, flow_feat_dim],      # float32
    "packet_features": Tensor[N, ...] or None,      # uint8
    "node_features": Tensor[N, ...] or None,         # float32
    "element_classes": list[str] (len N),
    "flow_ids": list[str] (len N),
    "tick": list[int] (len N),
}
```

Each `shards_index.jsonl` line:

```json
{"shard": "shard_000000.pt", "num_samples": 200, "tick_min": 0, "tick_max": 14,
 "label_histogram": {"doorlock": 120, "hakai": 80}, "file_size_bytes": 123456,
 "written_at": 1234567890.1, "flow_feat_shape": [200, 42],
 "packet_feat_shape": null, "node_feat_shape": null}
```

A shard is flushed when either `data_collection_shard_size` samples have
buffered, or `data_collection_flush_interval_secs` has elapsed since the
last flush. `TigerBrain.shutdown()` calls `data_recorder.close()`, which
forces a final flush so no in-flight samples are lost when an experiment is
stopped (`stop-experiment` in `dash_cli.py`).

### `manifest.json`

Built once by `TigerBrain._build_data_collection_manifest()` and written by
`FlowDataRecorder` at construction time. It is a snapshot of everything
needed to reconstruct an equivalent `TigerBrain`/`NewTigerEnvironment`
outside the live controller process:

```python
{
    "intrusion_detection": {...},   # full intrusion_detection cfg sub-dict
    "neural_modules": {...},
    "knowledge": {...},
    "rewards": {...},
    "health": {...},
    "wandb": {...},
    "traffic_dict": {...},
    "container_ips": {...},
    "ips_containers": {...},
    "use_packet_feats": bool,        # model config's flag, not the recorder's
    "use_node_feats": bool,          # model config's flag, not the recorder's
    "flow_feat_dim": int,
    "packet_feat_dim": int,
    "hidden_size": int,
    "device": "cpu" | "cuda:N",
    "models": "<exec'able python source defining the model classes>",
}
```

Note `manifest["use_packet_feats"]`/`["use_node_feats"]` reflect the
**model's** desire for those streams, which is not the same thing as
whether the recorder actually persisted them — that depends on
`data_collection_use_packet_feats`/`_node_feats` *also* being true at
capture time (see the coverage check in §2).

### Inspecting a run

`read_collected_data.py <data_collection_dir> [--latest] [--verify] [--show-sample]`
is a standalone tool (no POX/controller dependency) for sanity-checking a
collection run before trusting it:

- Default: lists every run directory found, with a one-line summary from
  `shards_index.jsonl` (shard count, total samples, total size, tick range,
  aggregated label histogram) — no shard loading.
- `--verify`: loads every shard and checks file existence, tensor
  shapes/dtypes against what the index claimed at write time, per-sample
  list lengths, and tick monotonicity within and across shards.
- `--show-sample`: prints one concrete sample's tensors/label for an eyeball
  check.

### Logging

Both `FlowDataRecorder` and the `TigerBrain` data-collection code path log
extensively and are designed to fail loud, not silent:

- Initialization logs `out_dir`, `shard_size`, `flush_interval_secs`, and
  the *effective* `use_packet_feats`/`use_node_feats` flags.
- Every `record()` call logs (at `debug`) the running per-shard label
  histogram; mismatches (e.g. sample-count vs. label-count) are logged at
  `error` and the whole batch is dropped rather than partially recorded.
- Every flush logs the shard name, sample count, file size, write latency,
  tick range, and label histogram.
- `close()` logs the final total samples/shards for the run.
- If `packet_features`/`node_features` are requested but turn out to be
  `None` partway through a run, a warning is logged and that feature stream
  is disabled for the *rest* of the run (to avoid ragged/inconsistent
  shapes within a shard).

---

## 2. Offline replay (`offline_replay.py`)

### Purpose

Replays a previously-recorded run through a `TigerBrain` instantiated
directly in a plain Python process — no FastAPI, no POX, no uvicorn, no
GNS3 topology. The recorded shards are fed through **the same code path**
the live controller used (`TigerBrain._process_batch`), so this is not a
re-implementation of training logic against the recorded tensors; it's the
original logic, fed recorded data instead of live flows.

### Code changes that enable this (in `tiger_brain_new.py`)

- **`_process_batch(self, batch)`** — extracted from `process_input`'s
  inner `with self._lock:` block. Contains: push to replay buffers →
  online inference (if `batch_processing_allowed`) → single-batch training
  step (re-checked, since knowledge can change during online inference) →
  step-counter increment. Called identically by `process_input` (online)
  and `process_input_from_record` (offline) — online behavior is
  unchanged, this is a pure extraction.
- **`get_labels_from_strings(self, labels)`** — extracted from `get_labels`.
  Takes raw NL strings directly (mutating the dynamic encoder / knowledge
  base exactly as before via `encoder.fit` + `add_class_to_knowledge_base`).
  `get_labels(self, flows)` is now a thin wrapper:
  `get_labels_from_strings([f.element_class for f in flows])`.
- **`process_input_from_record(self, flow_features, packet_features, node_features, element_classes, tick=None)`** —
  builds a `Batch` directly from already-recorded tensors, calls
  `get_labels_from_strings(element_classes)`, logs a summary (tick, sample
  count, label histogram, known-class count, `batch_processing_allowed`),
  then calls `_process_batch(batch)`.

### What `offline_replay.py` does, step by step

```
python3 offline_replay.py <run_dir> [options]
```

1. **Loads `manifest.json` and `shards_index.jsonl`** from `<run_dir>`.
2. **Feature coverage check** (fails loudly, does not proceed silently):
   compares what the model config wants (`manifest["use_packet_feats"]`/
   `["use_node_feats"]`) against what was actually captured
   (`manifest["intrusion_detection"]["data_collection_use_packet_feats"]`/
   `["data_collection_use_node_feats"]`). If the model expects a feature
   stream that wasn't captured, the script aborts with an explicit error
   rather than silently feeding `None` where the model expects a real
   tensor.
3. **Reconstructs `kwargs`** for `TigerBrain.__init__` from the manifest:
   `intrusion_detection`/`neural_modules`/`knowledge`/`rewards`/`health`/
   `wandb`/`traffic_dict`/`container_ips`/`ips_containers`/`device`/`models`,
   plus a real `logging.Logger`. Critical override (always logged):
   **`intrusion_detection.data_collection_mode` is forced to `False`** —
   replay must not spin up a second `FlowDataRecorder` and record itself.
4. **CLI overrides** (all logged when applied):
   - `--device` — override the manifest's recorded device.
   - `--pretrained-models-dir` — override where models load from/save to.
   - `--load-pretrained` — force `pretrained_inference=True`.
   - `--no-save` — disable checkpoint saving for this replay.
   - `--wandb` / `--wandb-run-name` — opt in to real (online) wandb
     tracking; **default is disabled** (a real `WandBTracker` is still
     constructed — `step_counter` and the resource-monitor thread are
     needed by `TigerBrain` regardless — just with `mode="disabled"`, so no
     network calls happen).
   - `--max-shards` / `--max-samples` — bound how much of the run is
     replayed (smoke-testing a large run).
5. **Constructs `WandBTracker(kwargs)`, then `TigerBrain(kwargs, wb_tracker=...)`**
   exactly as the live controller does in `tiger_server.py`'s `/initialize`
   handler.
6. **Replays shards in index order.** For each shard:
   - `torch.load`s it; on failure, logs the error and skips the *shard*
     (replay continues with the next one).
   - Groups samples into contiguous same-tick runs
     (`iter_tick_groups`). A single online `record()` call (one tick) is
     always buffered and flushed atomically, so a tick's samples are never
     split across two shards — grouping within each shard alone faithfully
     reconstructs the original per-tick batch boundaries.
   - For each tick-group, calls `brain.process_input_from_record(...)`. On
     a per-tick exception, logs it and continues with the next tick group
     (one bad batch doesn't abort the whole replay).
   - Logs per-shard completion: tick-batches replayed, samples replayed,
     elapsed time, and running totals (samples/ticks/shards, known-class
     count, `step_counter`, `batch_processing_allowed`).
7. **Final summary log**: total elapsed time, shards replayed vs. shard
   errors, ticks/samples replayed, final known-class count, final
   `step_counter`, aggregate label histogram across the whole replay.
8. **`brain.shutdown()`** — flushes wandb, stops the resource-monitor
   thread. (Model checkpointing itself is automatic — see below — not
   something this script triggers explicitly.)

Exit code is `0` on a clean replay, `2` if any shard failed to load/parse
(so the run is still mostly usable but should be investigated), `1` on a
hard precondition failure (missing manifest fields, feature coverage
mismatch, `TigerBrain` construction failure).

### Model checkpointing during replay

No new save logic was added. `process_input_from_record` → `_process_batch`
exercises the exact same `train_inf_module_single_batch` → periodic
`start_async_evaluation` → `check_progress_and_save` → `save_model` path
that online runs use, gated by the manifest's recorded
`intrusion_detection.online_evaluation`/`online_eval_step_freq` and
`save_models` flag (overridable with `--no-save`). Checkpoints land under
`pretrained_models_dir`, keyed by `wandb.wb_run_name`, same as online.

### Known limitation

If a collection run captured only `flow_features` (the common case —
`data_collection_use_packet_feats`/`_node_feats` both default `false`) but
the model config the manifest describes wants packet or node features,
replay cannot proceed faithfully and the script refuses to run (see the
feature coverage check in step 2). To replay with packet/node features,
the **original collection run** must be re-done with
`data_collection_use_packet_feats`/`_node_feats: true`.

---

## 3. Quick reference

| Task | Command |
|---|---|
| Capture a run (via dash_cli, see `tiger/tests/data_collection_test.py`) | `--profile data_collection init-config`, set `data_collection_dir`/`shard_size`, `start-experiment`/`start-traffic`, wait, `stop-experiment`/`stop-traffic` |
| List/verify a captured run | `python3 read_collected_data.py <data_collection_dir> --latest --verify` |
| Replay/train offline on a captured run | `python3 offline_replay.py <data_collection_dir>/run_<timestamp>` |
| Smoke-test a replay without committing to the whole run | add `--max-shards N` or `--max-samples N` |
