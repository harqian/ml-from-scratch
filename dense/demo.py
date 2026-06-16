"""renders demo.gif: the hand-derived matrix-backprop network learning real MNIST
(28x28), raced head-to-head against scikit-learn's MLPClassifier with the same
784->100->10 architecture.

left panel: held-out test accuracy per epoch, from-scratch vs sklearn.
right panel: live predictions from the from-scratch model, green=correct, red=wrong.

mirrors matrix_backprop_mnist.new_train_mnist (2000 train / 200 test, same lr
schedule). run: python3 demo.py"""

import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.neural_network import MLPClassifier
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from matrix_backprop_mnist import TestModel, one_hot_encode

TRAIN, TEST, EPOCHS, EVERY = 2000, 200, 300, 5


def load():
    (xi, yi), (xt, yt) = mnist.load_data()
    xi = xi.astype(np.float32)[:TRAIN] / 255.0
    xt = xt.astype(np.float32)[:TEST] / 255.0
    test_imgs, test_lbl = xt.copy(), yt[:TEST].copy()
    Xtr = xi.reshape(-1, 784).T
    Xte = xt.reshape(-1, 784).T
    ytr = one_hot_encode(yi[:TRAIN]).T
    yte = one_hot_encode(yt[:TEST]).T
    return Xtr, Xte, ytr, yte, test_imgs, test_lbl


def main():
    Xtr, Xte, ytr, yte, test_imgs, test_lbl = load()
    te_truth = np.argmax(yte.T, axis=1)

    # from-scratch model (hand-derived gradients)
    scratch = TestModel(784, 10)
    # sklearn reference: same hidden width + sigmoid activation, one pass per epoch
    skl = MLPClassifier(hidden_layer_sizes=(100,), activation="logistic",
                        solver="sgd", learning_rate_init=0.05,
                        warm_start=True, max_iter=1)
    skl_Xtr, skl_ytr = Xtr.T, np.argmax(ytr.T, axis=1)

    scratch_acc, skl_acc, xs, frames = [], [], [], []
    preds = None

    import warnings
    warnings.filterwarnings("ignore")

    for e in range(EPOCHS):
        lr = 0.002 if e < 150 else 0.001
        scratch.step(Xtr, ytr, lr)
        skl.fit(skl_Xtr, skl_ytr)

        if e % EVERY == 0 or e == EPOCHS - 1:
            preds = scratch(Xte).T
            sa = np.mean(np.argmax(preds, axis=1) == te_truth)
            ka = np.mean(skl.predict(Xte.T) == te_truth)
            scratch_acc.append(sa); skl_acc.append(ka); xs.append(e + 1)

            fig = plt.figure(figsize=(9.4, 4.3), dpi=100)
            gl = fig.add_subplot(1, 2, 1)
            gl.plot(xs, np.array(scratch_acc) * 100, color="#2563eb", lw=2.3,
                    marker="o", ms=3, label="from scratch (hand-derived)")
            gl.plot(xs, np.array(skl_acc) * 100, color="#9ca3af", lw=2.0,
                    ls="--", label="sklearn MLPClassifier")
            gl.set_xlim(0, EPOCHS); gl.set_ylim(0, 100)
            gl.set_title(f"held-out accuracy  (epoch {e+1}/{EPOCHS})")
            gl.set_xlabel("epoch"); gl.set_ylabel("test accuracy (%)")
            gl.grid(alpha=0.2); gl.legend(loc="lower right", fontsize=8)
            gl.text(0.03, 0.93, f"scratch {sa*100:.1f}%  |  sklearn {ka*100:.1f}%",
                    transform=gl.transAxes, fontsize=9, color="#111")

            gr = fig.add_subplot(1, 2, 2); gr.axis("off")
            gr.set_title("from-scratch predictions on held-out MNIST")
            for k in range(9):
                ax = fig.add_axes([0.55 + (k % 3) * 0.135, 0.60 - (k // 3) * 0.255, 0.12, 0.20])
                ax.imshow(test_imgs[k], cmap="gray_r")
                ax.set_xticks([]); ax.set_yticks([])
                p = int(np.argmax(preds[k]))
                ok = p == test_lbl[k]
                for s in ax.spines.values():
                    s.set_edgecolor("#16a34a" if ok else "#dc2626"); s.set_linewidth(2.5)
                ax.set_title(str(p), color="#16a34a" if ok else "#dc2626", fontsize=11, pad=2)

            fig.canvas.draw()
            frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3])
            plt.close(fig)

    frames += [frames[-1]] * 10
    imageio.mimsave("demo.gif", frames, duration=0.13, loop=0)
    print(f"wrote demo.gif ({len(frames)} frames) | "
          f"scratch {scratch_acc[-1]*100:.1f}%  sklearn {skl_acc[-1]*100:.1f}%")


if __name__ == "__main__":
    main()
