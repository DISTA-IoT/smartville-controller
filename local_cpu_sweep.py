"""
4-worker parallel driver for the CTI-uncertainty-threshold ablation.
Reference modes + fixed_threshold_cti threshold sweep, seeds 6 & 1,
120 train / 10 eval episodes (matches REAL_DATA_CALIBRATION.md).
"""

import itertools, json, os, subprocess, sys, time

HERE = ''
DATA = 'pre_recorded_data/'
EPISODES, EVAL = '120', '10'
SEEDS = [6, 1, 2, 3, 4]
THRESHOLDS = [1.0, 1.25, 1.5, 2.0]
LOGDIR = 'simba_sweeplogs'
os.makedirs(LOGDIR, exist_ok=True)

jobs = []   # (name, extra_args)
for m in ['drl', 'no_epistemic', 'greedy_cti']:
    for s in SEEDS:
        jobs.append((f'{m}_seed{s}', ['--mode', m, '--seed', str(s)]))
for th in THRESHOLDS:
    for s in SEEDS:
        jobs.append((f'fixed_threshold_cti_t{th}_seed{s}',
                     ['--mode', 'fixed_threshold_cti', '--seed', str(s),
                      '--set', f'cti_confidence_threshold={th}']))

env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
           OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
def cmd(name, extra):
    return [sys.executable, 'simba_offline.py', DATA, '--no-manifest', '--wandb-project', 'SIMBA',
            '--episodes', EPISODES, '--eval-episodes', EVAL,
            '--print-every', '40', '--run-name', name] + extra

WORKERS = 4
running = {}   # proc -> (name, log)
pending = list(jobs)
done = []
t0 = time.time()
print(f'[sweep] {len(jobs)} runs, {WORKERS} workers, 120/10 eps', flush=True)
while pending or running:
    while pending and len(running) < WORKERS:
        name, extra = pending.pop(0)
        log = open(os.path.join(LOGDIR, name + '.log'), 'w')
        p = subprocess.Popen(cmd(name, extra), env=env,
                             stdout=log, stderr=subprocess.STDOUT)
        running[p] = (name, log)
        print(f'[sweep] start {name}  ({len(done)} done, {len(pending)} queued)', flush=True)
    time.sleep(3)
    for p in [p for p in running if p.poll() is not None]:
        name, log = running.pop(p); log.close(); done.append(name)
        print(f'[sweep] DONE {name} rc={p.returncode}  (+{int(time.time()-t0)}s)', flush=True)
print(f'[sweep] all done in {int(time.time()-t0)}s', flush=True)

# aggregate
print('\n[sweep] ===== RESULTS =====', flush=True)
rows = []
for name, _ in jobs:
    path = os.path.join(HERE, 'runs_simba', name, 'results.json')
    if os.path.isfile(path):
        with open(path) as f: r = json.load(f)
        rows.append((name, r))
        print(f"  {name:34s} ret {r['eval_mean_return']:8.1f} ± {r['eval_std_return']:6.1f}  "
              f"buys {r['eval_mean_buys']:.1f}  {r['eval_buys_per_class']}", flush=True)
    else:
        print(f"  {name:34s} MISSING", flush=True)
with open(os.path.join(LOGDIR, 'agg.json'), 'w') as f:
    json.dump({n: r for n, r in rows}, f, indent=2)
print('[sweep] wrote agg.json', flush=True)