# Migrating `smartville-controller` from POX to Kytos-NG

**Audience:** an autonomous coding agent (or a developer) executing this migration.
**Scope:** replace POX (unmaintained) with **Kytos-NG** (`kytos-ng/kytos`, actively
maintained, pure Python, OpenFlow message library `python-openflow`/`pyof`) as the
southbound/event-loop layer, while preserving 100% of the NIDS/RL behavior.
**Non-goals (explicitly out of scope for this migration):**
- Do **not** touch `tiger_brain_new.py`, `tiger_agents.py`, `tiger_environment_new.py`,
  `neural_modules.py`, `replay_buffer.py`, `brain_utils.py`, `cti_delivery.py`,
  `data_recorder.py`, `im_models/*`. These have **zero POX imports** (verified by
  `grep -rl "pox\." --include="*.py" .` — only `entry.py`, `flowlogger_new.py`,
  `smart_switch.py`, `tiger_server.py` match) and must not change.
- Do **not** upgrade OpenFlow 1.0 → 1.3 in this pass. Kytos supports multi-version
  OF, so that upgrade becomes possible afterward, but bundling it here removes your
  ability to verify "did Kytos break something" vs "did OF1.3 break something"
  independently. Treat it as a **follow-up migration** once this one is merged.
- Do **not** start the P4/bmv2 migration yet. That is a separate, later phase
  (different dataplane entirely — see prior discussion). This document only gets
  you off POX onto a maintained OpenFlow controller library, still against the
  existing `gns3/openvswitch` switches.

---

## 1. Why this is tractable

POX usage is narrowly confined to **4 files** out of the whole codebase:

| File | POX imports |
|---|---|
| `entry.py` (47 lines) | `pox.openflow.libopenflow_01 as of` (only for `of.OFPP_NONE`) |
| `flowlogger_new.py` (227 lines) | `pox.core.core`, `pox.openflow.of_json.flow_stats_to_list`, `pox.lib.packet.ipv4.ipv4` |
| `smart_switch.py` (766 lines) | `pox.core.core`, `pox.lib.revent.EventMixin`, `pox.lib.recoco.Timer`, `pox.lib.packet.{ipv4,arp,ethernet}`, `pox.lib.addresses.EthAddr`, `pox.openflow.libopenflow_01 as of` |
| `tiger_server.py` (630 lines) | `pox.core.core`, `pox.openflow.libopenflow_01 as of`, `pox.lib.addresses.EthAddr` |

The NIDS/RL "brain" is fed plain `Flow` objects (`flow.py`) built by `flowlogger_new.py`
and never talks OpenFlow directly. Today it also does **not** enforce verdicts back
onto the switch (the only live flow-deletion call site,
`SmartSwitch.delete_ip_flow_matching_rules`, is triggered by ARP-table changes, not
by NIDS classification) — confirm this hasn't changed before you start, since if a
verdict→block wiring has been added since this doc was written, it changes Step 3's
blast radius.

This means the migration is a **bounded, mechanical port of 4 files**, not a rewrite
of the ML system.

---

## 2. POX → Kytos-NG API mapping

Use this table while porting. `pyof` = the `python-openflow` package (Kytos's
OpenFlow message library, versioned per OF version: `pyof.v0x01` = OF1.0).

| POX | Kytos-NG / pyof equivalent | Notes |
|---|---|---|
| `pox.core.core.getLogger()` | standard `logging.getLogger(__name__)` | Kytos NApps use stdlib logging via `self.log` (inherited from `KytosNApp`, or `from kytos.core import log`). |
| `core.register(name, component)` / `EventMixin` subclass | a `KytosNApp` subclass in `napps/<user>/<napp>/main.py` | One NApp per POX "component". `SmartSwitch` → `SmartSwitchNApp`. |
| `_handle_openflow_PacketIn(self, event)` | `@listen_to('kytos/of_core.v0x01.messages.in.ofpt_packet_in')` decorated method | Event name is version-specific; OF1.0 = `v0x01`. |
| `_handle_openflow_FlowRemoved(self, event)` | `@listen_to('kytos/of_core.v0x01.messages.in.ofpt_flow_removed')` | |
| `_handle_openflow_ConnectionUp` | `@listen_to('kytos/core.switch.new')` or `kytos/of_core.handshake.completed` | Check exact event name against the installed `kytos/of_core` NApp version — verify empirically in Step 1. |
| `_handle_openflow_ConnectionDown` | `@listen_to('kytos/core.switch.deleted')` (or `.disconnected`) | ditto |
| `pox.lib.recoco.Timer(secs, fn, recurring=True)` | `KytosNApp.execute()` + `self.execute_as_loop(interval)` called in `setup()`, **or** a plain `threading.Timer`/loop thread if you need multiple independent periods | Kytos gives each NApp exactly one `execute()` loop; `smart_switch.py` needs two independent periods (`_expire_timer` @5s, `sampling_rules_timer` @`sampling_rate_seconds`) — either run `execute()` at the GCD of the two periods and use internal counters, or spawn a second `threading.Timer`/daemon thread manually (simplest, lowest-risk port). |
| `pox.openflow.libopenflow_01.ofp_flow_mod(command=..., match=..., actions=..., priority=..., idle_timeout=..., hard_timeout=..., buffer_id=..., flags=...)` | `pyof.v0x01.controller2switch.flow_mod.FlowMod(...)` | Field names differ slightly — check `pyof` source for the exact constructor kwargs (e.g. `command`, `match`, `actions`, `priority`, `idle_timeout`, `hard_timeout`, `buffer_id`, `flags`). |
| `of.OFPFC_ADD` / `of.OFPFC_DELETE` | `pyof.v0x01.controller2switch.flow_mod.FlowModCommand.OFPFC_ADD` / `OFPFC_DELETE` | |
| `of.ofp_match(dl_type=..., nw_src=..., nw_dst=...)` | `pyof.v0x01.common.flow_match.Match(dl_type=..., nw_src=..., nw_dst=...)` | |
| `of.ofp_action_output(port=..., max_len=...)` | `pyof.v0x01.common.action.ActionOutput(port=..., max_length=...)` | Note `max_len` → `max_length`. |
| `of.ofp_action_dl_addr.set_dst(mac)` | `pyof.v0x01.common.action.ActionDLAddr(action_type=ActionType.OFPAT_SET_DL_DST, dl_addr=mac)` | |
| `of.OFPP_FLOOD` / `OFPP_CONTROLLER` / `OFPP_IN_PORT` / `OFPP_NONE` | `pyof.v0x01.common.phy_port.Port.OFPP_FLOOD` / `OFPP_CONTROLLER` / `OFPP_IN_PORT` / `OFPP_NONE` | |
| `of.OFP_FLOW_PERMANENT` | `0` (pyof uses the same OF-spec sentinel; confirm constant name, may just be a literal `0`) | |
| `of.OFPFF_SEND_FLOW_REM` | `pyof.v0x01.controller2switch.flow_mod.FlowModFlags.OFPFF_SEND_FLOW_REM` | |
| `of.ofp_packet_out(buffer_id=..., in_port=..., actions=[...], data=...)` | `pyof.v0x01.controller2switch.packet_out.PacketOut(...)` | |
| `of.ofp_stats_request(body=of.ofp_flow_stats_request())` / `ofp_port_stats_request()` | `pyof.v0x01.controller2switch.stats_request.StatsRequest(body_type=..., body=FlowStatsRequest())` / `PortStatsRequest()` | Consider using the `kytos/of_stats` NApp (ships with the Kytos ecosystem, already polls flow/port stats) instead of hand-rolling this — evaluate in Step 4. |
| `pox.openflow.of_json.flow_stats_to_list(event.stats)` | no direct equivalent — iterate the `pyof` `FlowStatsReply`/multipart reply objects and read fields directly (`.match.nw_src`, `.byte_count`, `.duration_sec`, `.duration_nsec`, `.packet_count`, `.actions`) | You are already only reading a handful of fields (see `flowlogger_new.process_received_flow`); write a small adapter function instead of trying to find a like-for-like JSON dumper. |
| `pox.lib.packet.ipv4`, `pox.lib.packet.arp`, `pox.lib.packet.ethernet` | `pyof.v0x01.common.action` doesn't parse L2/L3 — use Python's own `scapy` (already a transitive/available dep pattern in this project's sibling containers) or keep POX's own packet-parsing lib (`pox/lib/packet/*.py` is close to license-clean, self-contained parsing code with no other POX coupling — vendoring just this submodule is a legitimate low-risk option) | This is the one area with **no first-class Kytos replacement** — decide explicitly in Step 2 rather than discovering it mid-port. |
| `pox.lib.addresses.EthAddr` | Kytos ships `kytos.core.common.EntityStatus`-style helpers, but for a plain MAC type, Python's `str` + a tiny helper, or reuse `pyof.v0x01.common.phy_port`'s address handling | Low risk, small surface (`dpid_to_mac` in `smart_switch.py`/`tiger_server.py`, and `entry.py`'s `Entry.mac`). |
| `core.openflow._connections` (dict of dpid→connection) | `self.controller.switches` (dict of dpid→`Switch`), each with `.connection` | |
| `core.openflow.sendToDPID(dpid, msg)` | `switch = self.controller.get_switch_by_dpid(dpid); switch.connection.send(msg.pack())` | |
| `connection.send(msg)` / `connection.send(msg.pack())` | `switch.connection.send(msg.pack())` | Kytos, like POX, sends raw packed bytes over the socket. |
| `core.openflow.addListenerByName("FlowStatsReceived", fn)` / `.removeListener(fn)` | `@listen_to(...)` decorator is static (registered at NApp load, not dynamically per-experiment) | This is an **architectural mismatch** with `tiger_server.py`'s per-`/initialize` dynamic listener attach/detach — see Step 5. |
| FastAPI app run in a background thread inside the POX process | Same pattern still works (Kytos does not forbid embedding another web framework in a thread) — **or** migrate routes to Kytos's native `@rest` decorator on the NApp | Recommend keeping FastAPI as-is for this migration (Step 5) to minimize simultaneous changes; note native REST as a future cleanup. |

---

## 3. The one real architectural risk: concurrency model

POX's `recoco` scheduler is **cooperative and single-threaded** — every
`_handle_openflow_*` handler in `smart_switch.py` and every `Timer` callback runs on
the same thread, one at a time. This is why `smart_switch.py`'s mutable state
(`self.arpTables`, `self.unprocessed_flows`, `self.forwardingRules`,
`self.sampling_rule_messages`, `self.recently_sent_ARPs`) has **no locks anywhere**
— it never needed them.

Kytos-NG dispatches `@listen_to` handlers through its own event buffer/thread pool,
and NApp `execute()` loops run on their own thread. **This safety property disappears
under Kytos.** A `PacketIn` handler and a `FlowRemoved` handler (or your
`execute_as_loop` timer) can now run concurrently and race on the same dict.

**Action required:** wrap all of `SmartSwitch`'s (Kytos NApp's) shared mutable state
accesses in a single `threading.Lock` (mirroring the existing `tiger_lock` pattern
already used in `tiger_server.py`). Do this in Step 3, not as an afterthought — a race
here fails silently (corrupted ARP table / duplicate flow installs) rather than
crashing, so it's exactly the kind of bug that survives to a demo.

---

## 4. Step-by-step plan

Each step has a **goal**, **changes**, and a **verification gate** you must pass
before starting the next step. Do not batch steps — the whole point of doing this
stepwise is that when something breaks, you know which step broke it.

### Step 0 — Capture a baseline (no code changes)

**Goal:** have a ground-truth reference to diff every later step against.

**Do:**
1. On the current `new_smartville` branch (still POX-based), run the existing GNS3
   star topology (`utils/star_topology.py` in `insubria-smartville`) with a short,
   fixed traffic scenario (pick one deterministic benign + one attack pattern already
   used in the test fixtures).
2. Capture: the W&B run (byte/packet counts per flow, `flow_creates`/`flow_expires`
   counters, `dropped_packets`), and — separately — a raw `tcpdump`/pcap of the
   controller↔switch OpenFlow channel if easy to obtain (helps debug wire-format
   diffs later).
3. Save these as a fixture (e.g. `docs/migration_baseline/`), not just logs someone
   has to dig out of a dashboard.

**Verify:** the baseline artifact exists and is committed. This step produces no
functional change, so "verification" is just "the fixture is captured and
reproducible."

---

### Step 1 — Stand up Kytos-NG in isolation

**Goal:** prove Kytos-NG can complete an OF1.0 handshake with your actual switch
image, with zero SmartVille logic involved yet. Isolates infrastructure risk from
logic-port risk.

**Do:**
1. In a scratch branch, add `kytos` + `python-openflow` to a **new**, parallel
   requirements file (don't touch `requirements.txt` yet).
2. Boot `kytosd` (Kytos's daemon) locally or in a throwaway container, using its
   stock `kytos/of_core` NApp (ships with Kytos, handles the OF handshake/echo —
   you get this for free, POX's `openflow.discovery`/handshake logic has no
   equivalent code to port).
3. Point **one** `gns3/openvswitch` instance's controller config
   (`openSwitch/boot.sh` in `insubria-smartville`, currently
   `192.168.1.1:6633`) at the Kytos listener instead (Kytos's default OF port is
   `6653`, but it's configurable — either change the switch or configure Kytos to
   listen on `6633` to avoid touching the switch image at all).
4. Confirm via Kytos's REST API (`GET /api/kytos/core/switches`) that the switch
   shows up as connected, with the DPID matching what POX used to see in its logs.

**Verify:** `GET /api/kytos/core/switches` returns your switch's DPID as connected.
No SmartVille code has been touched — if this fails, the problem is purely
Kytos/network config, not your port.

---

### Step 2 — Port the low-level primitive layer (`entry.py`)

**Goal:** smallest possible file first, and resolve the "no first-class L2/L3 packet
parsing in Kytos" question (see mapping table §2) before it blocks a bigger file.

**Do:**
1. Port `entry.py`'s single POX dependency (`of.OFPP_NONE`) to
   `pyof.v0x01.common.phy_port.Port.OFPP_NONE`.
2. Decide and document (in this file, amending §2) how you'll handle
   `pox.lib.packet.{ipv4,arp,ethernet}` and `pox.lib.addresses.EthAddr` for Step 3 —
   either vendor POX's `pox/lib/packet/*.py` parsing modules (they have no other POX
   coupling) or adopt `scapy`.

**Verify:** write a standalone unit test (no live switch needed) that imports the
ported `entry.py` and asserts `Entry(...).isExpired()` behaves identically to before
for a few port/timeout combinations, including the `port == OFPP_NONE` special case.
This is a pure logic test — if it doesn't pass without any network, don't proceed.

---

### Step 3 — Port `smart_switch.py` into a Kytos NApp

**Goal:** the biggest, highest-risk file. Port `SmartSwitch` → a
`napps/dista_iot/smart_switch/main.py` `KytosNApp`, applying every row of the
mapping table in §2, **plus the locking from §3**.

**Do (in this order, each individually testable):**
1. Scaffold the NApp (`kytos.json`, `main.py`, `settings.py`) with an empty
   `setup()`/`execute()`/`shutdown()` — get it loading under `kytosd` before porting
   any logic.
2. Port `ForwardingRule` and the ARP-table (`Entry`-based) bookkeeping verbatim
   (pure Python, no POX dependency) — add the lock from §3 around every dict access.
3. Port `_handle_openflow_PacketIn` → the `@listen_to('...ofpt_packet_in')` method,
   including the `OFPR_ACTION` sampling short-circuit (this is the packet-mirroring
   path feeding the NIDS — do not lose the "cache-only, don't touch routing" branch).
4. Port ARP handling (`handle_arp_packet_in`, `send_arp_response`,
   `build_and_send_ARP_request`) using whichever packet-parsing approach you chose
   in Step 2.
5. Port IPv4 handling (`handle_ipv4_packet_in`, `try_creating_flow_rule`,
   `add_ip_to_ip_flow_matching_rule` — **including** the dual flow-mod install
   (forwarding rule *and* the higher-priority sampling rule with
   `OFPP_CONTROLLER`/`max_length`) and `delete_ip_flow_matching_rules`).
6. Port `_handle_openflow_FlowRemoved` and the two timers
   (`_handle_expiration` @5s, `send_sampling_rules` @`sampling_rate_seconds`) per
   the Timer row in §2.
7. Port `pause()`/re-`initialize()` semantics — Kytos NApps are normally
   singleton-loaded once at `kytosd` startup, not re-constructed per HTTP call the
   way `tiger_server.py` does today (`core.hasComponent("smart_switch")` check).
   Keep the NApp instance alive and expose an explicit `reinitialize(**kwargs)`
   method that `tiger_server.py` calls, matching current behavior — do not try to
   unload/reload the NApp itself.

**Verify:** build a **minimal 2-host, 1-switch** GNS3 (or plain Mininet/manual OVS)
topology — deliberately smaller than the full star topology so iteration is fast.
Script: host A pings host B. Confirm, compared against Step 0's baseline semantics:
- ARP resolves (both directions) and stays cached until `arp_timeout`.
- A forwarding flow-mod is installed on first packet, a sampling flow-mod is
  installed alongside it at priority 200.
- Unplugging/replugging (or `docker restart`-ing) the switch mid-test correctly
  triggers `ConnectionUp`/`ConnectionDown` handling.
- `FlowRemoved` events correctly evict the local `forwardingRules` cache (test by
  waiting out `flow_idle_timeout` and confirming a fresh flow-mod gets reinstalled
  rather than silently dropping traffic).

Do not proceed to Step 4 until this 2-host smoke test is stable across at least a
few repeated runs (races from §3 tend to be intermittent).

---

### Step 4 — Port `flowlogger_new.py`

**Goal:** port the flow-stats polling → `Flow` feature-tensor pipeline.

**Do:**
1. Decide: hand-roll the stats-request timer (port `periodically_requests_stats`
   from `tiger_server.py` using `pyof`'s `StatsRequest`/`FlowStatsRequest`/
   `PortStatsRequest`, per §2), or adopt the `kytos/of_stats` NApp if it covers your
   needs (it ships in the standard Kytos NApp set — check whether its poll interval
   and reply granularity match what `process_received_flow` needs, i.e. per-flow
   `nw_src`/`nw_dst`/`byte_count`/`duration_sec`/`duration_nsec`/`packet_count`/
   `actions[1]['port']`). Hand-rolling is lower-risk/more predictable for this
   migration; note the NApp as a future simplification.
2. Port `_handle_flowstats_received`/`process_received_flow` to read fields off the
   `pyof` reply objects instead of POX's `flow_stats_to_list` dict-of-dicts shape.
3. Port `cache_unprocessed_packets`/`build_packet_tensor` — these only touch
   `packet.raw`/`packet.next`, i.e. whatever packet-parsing library you picked in
   Step 2, not POX's core/of module. Should be close to a no-op change.

**Verify:** write an integration test (can reuse the Step 3 2-host topology) that
runs a scripted, deterministic traffic burst and asserts the resulting `Flow`
objects' `byte_count`/`packet_count`/`duration_*` and packet-feature tensors match
Step 0's baseline within tolerance (exact byte-for-byte for the anonymized packet
tensor, since that logic is untouched; flow stats within normal timing jitter).

---

### Step 5 — Port `tiger_server.py`'s orchestration layer

**Goal:** wire the FastAPI control plane (`/initialize`, `/stop`, `/health`, etc.)
to the now-Kytos-based `SmartSwitch`/`FlowLogger`, preserving the re-initializable,
multi-experiment-per-process lifecycle.

**Do:**
1. Keep FastAPI running in a background thread exactly as today — do **not**
   migrate to Kytos's native `@rest` decorator in this pass (that's a separable
   cleanup, and mixing it in here reintroduces the "which change broke this"
   problem from the intro).
2. Replace `core.hasComponent`/`core.register`/`core.components[...]` access with
   direct references to your NApp singleton (Kytos exposes loaded NApps via
   `controller.napps[(username, napp_name)]`, or simpler: have `tiger_server.py`
   hold a module-level reference obtained at `kytosd`+FastAPI co-startup).
3. Replace the dynamic `core.openflow.addListenerByName("FlowStatsReceived", ...)`/
   `removeListener` pattern (used so each `/initialize` call rebinds the listener
   with fresh `traffic_dict`/`ips_containers` closures) — since `@listen_to` is
   static in Kytos, instead give the NApp a `set_context(traffic_dict, ips_containers,
   current_knowledge_provider)` method that `/initialize` calls, and have the
   `@listen_to`-decorated handler read from `self._context` (guarded by the §3
   lock) rather than being re-registered each time. This preserves exact
   behavior (fresh context per experiment) without depending on dynamic
   listener attach/detach.
4. Port `_flush_switch_flow_tables`-style enumeration (`core.openflow._connections`)
   to `self.controller.switches` per §2.
5. Leave `tiger_lock`, `smart_check()`, `periodically_requests_stats` threading
   structure as-is — they don't touch POX directly except through the objects
   you've already ported.

**Verify:** run the full `/initialize` → traffic → `/stop` → `/initialize` (again)
cycle at least twice in a row without restarting the process, confirming:
- Second `/initialize` doesn't double-register listeners (would show up as
  duplicate/doubled flow feature updates).
- `/health` still correctly reports crash state if you force an exception in
  `smart_check()`.
- `/pending_packet_feats_stats` still returns sane data mid-run.

---

### Step 6 — Full-topology regression test

**Goal:** acceptance gate for merging.

**Do:** run the *original* full GNS3 star topology with the *same* traffic scenario
captured in Step 0, end-to-end, on the Kytos-based controller.

**Verify (diff against Step 0's fixture):**
- Same set of flows created (by IP pair), same rough `flow_creates`/`flow_expires`
  counts (allow for OpenFlow timing jitter, not for structurally different counts).
- Same `dropped_packets` order of magnitude.
- Packet-sampling coverage (fraction of flows with non-empty
  `pending_packet_feats`/packet tensors) comparable to baseline.
- `controller_brain.process_input` runs without new exceptions over the full
  scenario duration (watch `/health`).
- No new W&B alert emails/Slack messages compared to baseline (i.e., you haven't
  introduced a new crash/disconnect pattern).

Only after this passes should `new_smartville` (or a PR into it) actually switch its
default southbound library.

---

### Step 7 — Cleanup / packaging

**Goal:** remove POX entirely from the build.

**Do:**
1. `controller.Dockerfile`: delete the `git clone .../pox` step; install `kytos` +
   `python-openflow` via `requirements.txt` instead; update `WORKDIR`/NApp install
   path (Kytos NApps typically live under `~/.kytos/napps/<user>/<napp>` or get
   symlinked in via `kytos napps install`/`kytos napps enable` — adapt the
   Dockerfile's `COPY`/`git clone smartville-controller` step accordingly instead of
   nesting inside a POX checkout).
2. `entrypoint.sh`: replace the (currently unused/commented-out) `pox.py` launch
   line with `kytosd` startup (foreground or via the existing FastAPI-thread
   pattern — `kytosd` itself should be the long-running foreground process now,
   analogous to how `../pox.py smartController.tiger_server.py` was before).
3. `requirements.txt`: add `kytos`, `python-openflow` (and `scapy` if chosen in
   Step 2); nothing to *remove* here since POX itself was git-cloned, not pip
   installed.
4. Delete now-dead POX-specific comments/docstrings referencing POX (e.g.
   `smart_switch.py`'s header pointing at `CPqD/RouteFlow`'s POX-based
   `l3_learning.py` — update to describe the Kytos NApp instead, or keep as
   historical-lineage context, your call, but don't leave it implying POX is still
   in use).
5. Update `insubria-smartville`'s `readme.md` (separate repo) — it currently states
   the controller container runs "the POX library" — cross-reference PR, don't fold
   into this repo's changes.

**Verify:** fresh `docker build` of `controller.Dockerfile` succeeds with zero POX
references (`grep -ri pox` inside the built image's `/pox`-equivalent working dir
returns nothing beyond incidental substring matches you've reviewed), and Step 6's
full-topology test still passes when run against the freshly built image (not just
your dev environment).

---

## 5. Suggested repo layout after migration

```
smartville-controller/
├── napps/
│   └── dista_iot/
│       └── smart_switch/
│           ├── kytos.json
│           ├── main.py          # ported smart_switch.py
│           ├── settings.py
│           └── entry.py         # ported, or inlined
├── flowlogger_new.py            # ported, stays a plain module used by the NApp + tiger_server
├── tiger_server.py              # ported orchestration/FastAPI layer
├── tiger_brain_new.py           # UNCHANGED
├── tiger_agents.py              # UNCHANGED
├── tiger_environment_new.py     # UNCHANGED
├── neural_modules.py            # UNCHANGED
├── ...                          # everything else UNCHANGED
├── docs/
│   ├── KYTOS_MIGRATION.md       # this file
│   └── migration_baseline/      # Step 0 fixtures
├── controller.Dockerfile        # updated in Step 7
├── entrypoint.sh                # updated in Step 7
└── requirements.txt             # updated in Step 7
```

## 6. Branch/PR strategy

Do each step (or small groups of adjacent low-risk steps, e.g. 0+1, or 6+7) as its
own commit or PR against a dedicated migration branch, **not** directly on
`new_smartville`. Merge into `new_smartville` only after Step 6 passes. This keeps
`git bisect` meaningful if the full-topology test regresses something subtle weeks
later.
