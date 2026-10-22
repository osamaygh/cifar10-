# Given by the lecturer

The starting point for Task 3, exactly as handed out.

| File | What it is |
|---|---|
| `colab_img_clf.ipynb` | The course notebook. Builds a small ConvNet for CIFAR-10, plus an EfficientNet-B0 transfer-learning model, and trains for one epoch. |

Two further items were supplied but are not stored here:

- **`cifar-10-batches-py/`** — the CIFAR-10 python batch files. Too large for a repository, and freely available from <https://www.cs.toronto.edu/~kriz/cifar.html>. All my results use these original files.
- **`modelz.pth`** — the lecturer's trained EfficientNet checkpoint (16 MB). Evaluated through the notebook's own pipeline it scores **76.24 %** on `test_batch`; `02-my-work/src/check_effnet.py` reproduces that number if you have the file.

Running this notebook unchanged gives about 10 % test accuracy, which is chance for ten classes. Finding out why is where my work starts.
