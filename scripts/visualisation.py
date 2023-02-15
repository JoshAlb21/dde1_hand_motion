import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

def plot_cm(y_true, y_pred, filename, class_labels, suffix_name:str):
    
    cf_matrix = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5])
    
    group_counts = ['{0:0.0f}'.format(value) for value in
                cf_matrix.flatten()]
    #normalize
    cf_matrix = (cf_matrix.T/cf_matrix.sum(axis=1)).T
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten()]
    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(5,5)

    custom_cmap = sns.light_palette("#009682", as_cmap=True)
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap=custom_cmap, xticklabels=class_labels, yticklabels=class_labels, cbar=False)
    plt.ylabel('wahre Klasse')
    plt.xlabel('geschätzte Klasse')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    save_path = os.path.join(filename, f'confusion_matrix_{suffix_name}.png')
    plt.savefig(save_path, bbox_inches = "tight", dpi=350)