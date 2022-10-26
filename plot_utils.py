import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve


def generate_roc_plot(y_score, y_true, dataset_name, model_name, results_dir):
    fpr, tpr, _ = roc_curve(y_true, np.array(y_score)[:, 1])
    plt.plot(fpr, tpr)
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.savefig(os.path.join(results_dir, f'{dataset_name}_{model_name}.png'))
    plt.clf()