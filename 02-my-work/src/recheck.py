"""Reload saved best checkpoints (results/<run>/best.pth) and re-test them on the 10k test set.
Usage: CIFAR_DIR=data/cifar-10-batches-py python src/recheck.py [results_dir]"""
import json, glob, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import ConvNet, ConvNetX, load
res = sys.argv[1] if len(sys.argv) > 1 else 'results'
cache = {}
for f in sorted(glob.glob(os.path.join(res, '*', 'metrics.json'))):
    r = json.load(open(f)); d = os.path.dirname(f)
    if not os.path.exists(os.path.join(d, 'best.pth')): continue
    ro = r.get('red_only', False)
    if ro not in cache: cache[ro] = load(['test_batch'], ro)
    x, y = cache[ro]
    m = ConvNet() if r['cfg'] == 'orig' else ConvNetX(**r['cfg'])
    m.load_state_dict(torch.load(os.path.join(d, 'best.pth'), map_location='cpu')); m.eval()
    with torch.no_grad():
        p = torch.cat([m(x[i:i + 1000].float()).argmax(1) for i in range(0, len(x), 1000)])
    acc = (p == y).float().mean().item() * 100
    print(f"{r['name']:18s} reported {r['best']['acc'] * 100:6.2f}  re-test {acc:6.2f}")
