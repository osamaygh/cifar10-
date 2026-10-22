"""Evaluate the course's EffNet_clf weights (modelz.pth) on the test set with the
notebook's exact pipeline (myData red-only bug, Normalize applied to 0-255 values).
Usage: CIFAR_DIR=data/cifar-10-batches-py python src/check_effnet.py path/to/modelz.pth
On the original CIFAR-10 test_batch this checkpoint scores 0.7624 (red-only pipeline),
against the 0.2792 printed in the course notebook's own one-epoch run."""
import os, sys, torch, torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import load

torch.set_num_threads(2)


class EffNet_clf(nn.Module):  # notebook class, but weights=None (no download); weights come from modelz.pth
    def __init__(self):
        super().__init__()
        self.rgb_normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.RGB_encoder = models.efficientnet_b0(weights=None)
        self.RGB_encoder.classifier[1] = nn.Linear(1280, 10)
        self.flatten = nn.Flatten()
        self.sigm = nn.Sigmoid()

    def forward(self, rgb):
        rgb = self.rgb_normalizer(rgb)
        f = self.RGB_encoder.features(rgb)
        return self.sigm(self.RGB_encoder.classifier(self.flatten(self.RGB_encoder.avgpool(f))))


m = EffNet_clf()
ckpt = sys.argv[1] if len(sys.argv) > 1 else 'modelz.pth'
sd = torch.load(ckpt, map_location='cpu', weights_only=True)
print(m.load_state_dict(sd))
m.eval()
for red_only in (True, False):
    x, y = load(['test_batch'], red_only=red_only)
    cor = 0
    with torch.no_grad():
        for i in range(0, len(x), 200):
            cor += (m(x[i:i + 200].float()).argmax(1) == y[i:i + 200]).sum().item()
    print(f'red_only={red_only}: test acc = {cor / len(x):.4f}')
print('params', sum(p.numel() for p in m.parameters()))
