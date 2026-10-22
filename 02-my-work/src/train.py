"""ConvNet architecture experiments on CIFAR-10.

The training protocol copies the course notebook (colab_img_clf.ipynb):
  * train = data_batch_1,3,5 (30k)  val = data_batch_2,4 (20k)  test = test_batch (10k)
  * input x/255, batch size 10, SGD(lr=0.0025, momentum=0.9)
  * Sigmoid output + BCE loss on one-hot labels (unless out='softmax')
Only the architecture changes between runs (plus number of epochs, which is
fixed for all runs and longer than the notebook's single epoch).
"""
import argparse, json, os, pickle, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.environ.get('CIFAR_DIR', 'data/cifar-10-batches-py') + '/'
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']


# ----------------------------------------------------------------------------- data
def unpickle(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')


def load(files, red_only=False):
    xs, ys = [], []
    for f in files:
        d = unpickle(ROOT + f)
        k = list(d.keys())
        ys.append(np.asarray(d[k[1]]))
        xs.append(d[k[2]])
    x = np.concatenate(xs).reshape(-1, 3, 32, 32)       # R,G,B planes -> N x C x H x W
    if red_only:                                        # reproduces the notebook's myData bug:
        x = np.repeat(x[:, :1], 3, axis=1)              # img_red/green/blue all use [0:1024]
    return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.concatenate(ys)).long()


# ----------------------------------------------------------------------------- models
class ConvNet(nn.Module):
    """Original model from the notebook, verbatim."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.flatten = nn.Flatten()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.globalpool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x = x / 255
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.flatten(self.globalpool(x))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigm(self.fc3(x))
        return x


class ConvNetX(nn.Module):
    """Configurable version of ConvNet. Defaults reproduce the original exactly."""
    def __init__(self, chans=(16, 32, 64), k=3, pad=0, convs=1, bn=False, pool='max',
                 head='gap', fc=(128, 64), dropout=0.0, out='sigmoid', act='relu'):
        super().__init__()
        A = {'relu': nn.ReLU, 'leaky': lambda: nn.LeakyReLU(0.1), 'elu': nn.ELU}[act]
        layers, c_in, s = [], 3, 32
        for c in chans:
            for j in range(convs):
                # pool='stride': the last conv of each block uses stride 2 and replaces the pooling layer
                st = 2 if (pool == 'stride' and j == convs - 1) else 1
                layers.append(nn.Conv2d(c_in, c, k, padding=pad, stride=st))
                s = (s + 2 * pad - k) // st + 1
                if bn:
                    layers.append(nn.BatchNorm2d(c))
                layers.append(A())
                c_in = c
            if pool in ('max', 'avg'):
                layers.append(nn.MaxPool2d(2) if pool == 'max' else nn.AvgPool2d(2))
                s //= 2
            assert s >= 1, 'feature map vanished - add padding or remove a block'
        self.features = nn.Sequential(*layers)
        self.gpool = nn.AdaptiveAvgPool2d(1) if head == 'gap' else nn.Identity()
        d_in = c_in if head == 'gap' else c_in * s * s
        cls = [nn.Flatten()]
        for d in fc:
            if dropout > 0:
                cls.append(nn.Dropout(dropout))
            cls += [nn.Linear(d_in, d), A()]
            d_in = d
        if dropout > 0:
            cls.append(nn.Dropout(dropout))
        cls.append(nn.Linear(d_in, 10))
        self.classifier = nn.Sequential(*cls)
        self.out = out
        self.final_hw = s

    def forward(self, x):
        x = x / 255
        x = self.classifier(self.gpool(self.features(x)))
        return torch.sigmoid(x) if self.out == 'sigmoid' else x   # raw logits for softmax+CE


def count_macs(model):
    """Multiply-accumulate ops for one 3x32x32 image (conv + linear layers)."""
    macs = []
    def hook(m, i, o):
        if isinstance(m, nn.Conv2d):
            macs.append(o.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1])
        elif isinstance(m, nn.Linear):
            macs.append(m.in_features * m.out_features)
    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, 32, 32))
    for h in hs:
        h.remove()
    return int(sum(macs))


# ----------------------------------------------------------------------------- train / eval
def loss_fn(out, y, kind):
    if kind == 'sigmoid':
        return F.binary_cross_entropy(out, F.one_hot(y, 10).float(), reduction='mean')
    return F.cross_entropy(out, y)


@torch.no_grad()
def evaluate(model, x, y, kind, bs=1000):
    model.eval()
    tot, preds = 0.0, []
    for i in range(0, len(x), bs):
        out = model(x[i:i + bs].float())
        tot += loss_fn(out, y[i:i + bs], kind).item() * len(out)
        preds.append(out.argmax(1))
    p = torch.cat(preds)
    return tot / len(x), (p == y).float().mean().item(), p


def metrics(p, y):
    cm = np.zeros((10, 10), dtype=int)
    np.add.at(cm, (y.numpy(), p.numpy()), 1)
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return {'acc': float(tp.sum() / cm.sum()), 'macro_f1': float(f1.mean()),
            'per_class_acc': rec.tolist(), 'per_class_f1': f1.tolist(), 'confusion': cm.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--cfg', default='{}', help='JSON kwargs for ConvNetX; "orig" = notebook class')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--red_only', action='store_true', help='reproduce notebook channel bug')
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--out', default='results')
    ap.add_argument('--aug', action='store_true', help='random crop (pad 4) + horizontal flip')
    ap.add_argument('--cosine', action='store_true', help='cosine learning-rate schedule')
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    xtr, ytr = load(['data_batch_1', 'data_batch_3', 'data_batch_5'], a.red_only)
    xva, yva = load(['data_batch_2', 'data_batch_4'], a.red_only)
    xte, yte = load(['test_batch'], a.red_only)

    if a.cfg == 'orig':
        model, cfg, kind = ConvNet(), 'orig', 'sigmoid'
    else:
        cfg = json.loads(a.cfg)
        model = ConvNetX(**cfg)
        kind = cfg.get('out', 'sigmoid')
    n_params = sum(p.numel() for p in model.parameters())
    macs = count_macs(model)

    opt = torch.optim.SGD(model.parameters(), lr=0.0025, momentum=0.9)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs) if a.cosine else None
    g = torch.Generator().manual_seed(a.seed)
    bs = 10
    run_dir = os.path.join(a.out, a.name)
    os.makedirs(run_dir, exist_ok=True)
    log = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
           'test_acc': [], 'epoch_time_s': []}
    best = {'val_loss': 1e9}
    for ep in range(a.epochs):
        model.train()
        t0 = time.time()
        perm = torch.randperm(len(xtr), generator=g)
        s_loss, s_cor, n = 0.0, 0, 0
        for i in range(0, len(xtr), bs):
            idx = perm[i:i + bs]
            xb, yb = xtr[idx].float(), ytr[idx]
            if a.aug:
                xp = F.pad(xb, (4, 4, 4, 4))
                ox, oy = torch.randint(0, 9, (2, len(xb)), generator=g)
                xb = torch.stack([xp[j, :, oy[j]:oy[j] + 32, ox[j]:ox[j] + 32] for j in range(len(xb))])
                flip = torch.rand(len(xb), generator=g) < 0.5
                xb[flip] = xb[flip].flip(3)
            out = model(xb)
            loss = loss_fn(out, yb, kind)
            opt.zero_grad()
            loss.backward()
            opt.step()
            s_loss += loss.item() * len(yb)
            s_cor += (out.argmax(1) == yb).sum().item()
            n += len(yb)
        t_ep = time.time() - t0
        if sched is not None:
            sched.step()
        vl, va, _ = evaluate(model, xva, yva, kind)
        tl, ta, tp = evaluate(model, xte, yte, kind)
        for k, v in zip(log, [ep + 1, s_loss / n, s_cor / n, vl, va, ta, t_ep]):
            log[k].append(v)
        if vl < best['val_loss']:
            best = {'val_loss': vl, 'epoch': ep + 1, 'val_acc': va, **metrics(tp, yte)}
            torch.save(model.state_dict(), os.path.join(run_dir, 'best.pth'))
        print(f'[{a.name}] ep {ep + 1:02d} train_loss {s_loss / n:.4f} train_acc {s_cor / n:.4f} '
              f'val_loss {vl:.4f} val_acc {va:.4f} test_acc {ta:.4f} ({t_ep:.0f}s)', flush=True)
        json.dump({'name': a.name, 'cfg': cfg, 'seed': a.seed, 'red_only': a.red_only, 'aug': a.aug, 'cosine': a.cosine,
                   'epochs': a.epochs, 'params': n_params, 'macs': macs, 'log': log, 'best': best},
                  open(os.path.join(run_dir, 'metrics.json'), 'w'), indent=1)

    # inference latency like the notebook's test loop (batch size 1), on the best checkpoint
    model.load_state_dict(torch.load(os.path.join(run_dir, 'best.pth')))
    model.eval()
    with torch.no_grad():
        for i in range(50):
            model(xte[i:i + 1].float())
        t0 = time.time()
        for i in range(1000):
            model(xte[i:i + 1].float())
        ms = (time.time() - t0)
    res = json.load(open(os.path.join(run_dir, 'metrics.json')))
    res['infer_ms_per_img'] = ms
    res['final_epoch'] = {k: v[-1] for k, v in log.items()}
    json.dump(res, open(os.path.join(run_dir, 'metrics.json'), 'w'), indent=1)
    print(f'[{a.name}] DONE best_ep {best["epoch"]} test_acc {best["acc"]:.4f} f1 {best["macro_f1"]:.4f} '
          f'params {n_params} macs {macs} infer {ms:.3f} ms/img', flush=True)


if __name__ == '__main__':
    main()
