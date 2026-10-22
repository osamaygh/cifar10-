"""Run every experiment listed in experiments.json (one after another).
Usage: python src/run_all.py [experiments.json] [results_dir]"""
import json, subprocess, sys, os
here = os.path.dirname(os.path.abspath(__file__))
jobs = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, '..', 'experiments.json')))
out = sys.argv[2] if len(sys.argv) > 2 else 'results'
for j in jobs:
    cmd = [sys.executable, os.path.join(here, 'train.py'), '--name', j['name'],
           '--cfg', j['cfg'] if isinstance(j['cfg'], str) else json.dumps(j['cfg']),
           '--epochs', str(j.get('epochs', 10)), '--seed', str(j.get('seed', 0)), '--out', out]
    cmd += ['--aug'] if j.get('aug') else []
    cmd += ['--cosine'] if j.get('cosine') else []
    cmd += ['--red_only'] if j.get('red_only') else []
    print('>>', j['name'], flush=True)
    subprocess.run(cmd, check=True)
