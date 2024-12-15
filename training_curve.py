import json
import glob
from dataclasses import dataclass
import numpy as np


@dataclass
class TrainingCurve:
    loss: np.ndarray
    acc: np.ndarray
    name: str

    def x(self):
        """
        Return the x-axis for the plot, scaled from 0 to 1
        """

        return np.linspace(0, 1, len(self.loss))



def load_training_curve(directory: str, name: str | None = None) -> TrainingCurve:
    loss_curve = []
    acc_curve = []

    for file in sorted(glob.glob(directory + "/*.json")):
        with open(file) as f:
            data = json.load(f)

            loss_curve += [i[0] for i in data]
            acc_curve += [i[1] for i in data]

    return TrainingCurve(np.array(loss_curve), np.array(acc_curve), name or directory)


b512_lr5en5 = load_training_curve("./results/b_512_lr_e-5")
b256_lr5en5 = load_training_curve("./results/b_256_lr_e-5")
b512_lreen4 = load_training_curve("./results/b_512_lr_e-4")
b256_lreen4 = load_training_curve("./results/b_256_lr_e-4")

import matplotlib.pyplot as plt

def moving_avg(x, w):
    x = x.copy()
    y = np.zeros_like(x)
    for i in range(1, len(x)):
        kern = x[max(0, i-w):i]
        y[i] = sum(kern) / len(kern)
    return y


b512_lr5en5.acc = moving_avg(b512_lr5en5.acc, 10)
b256_lr5en5.acc = moving_avg(b256_lr5en5.acc, 10)
b512_lreen4.acc = moving_avg(b512_lreen4.acc, 10)
b256_lreen4.acc = moving_avg(b256_lreen4.acc, 10)

plt.plot(b512_lr5en5.x(), b512_lr5en5.acc, label="Batch size 512, lr e-5")
plt.plot(b256_lr5en5.x(), b256_lr5en5.acc, label="Batch size 256, lr e-5")
plt.plot(b512_lreen4.x(), b512_lreen4.acc, label="Batch size 512, lr e-4")
plt.plot(b256_lreen4.x(), b256_lreen4.acc, label="Batch size 256, lr e-4")

plt.xlabel('Epoch')
plt.ylabel('Batch Accuracy')

plt.legend()

plt.show()