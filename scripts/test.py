import numpy as np
import pandas as pd

# Assume you have a DataFrame called 'df' with the columns ['X0', 'Y0', 'Z0', 'X1', 'Y1', 'Z1', 'X2', 'Y2', 'Z2']
# Write the columns as list

df_aggregate = pd.DataFrame({'Xtest': [33, 99]})

cols = ['X0', 'Y0', 'Z0', 'X1', 'Y1', 'Z1', 'X2', 'Y2', 'Z2']

vals = [[54.263880,	71.466776,	-64.807709,	76.895635,	42.462500,	-72.780545,	36.621229,	81.680557,	-52.919272], [1, 2,	3,	4, 5, 6, 7, 8, 9]]
# Create a DataFrame
df_test = pd.DataFrame(vals, columns=cols)
print('Original df:', df_test)

# Create an empty list to store the batches of data
batches = []

# Iterate over the DataFrame and extract the batches of data
for row in range(0, df_test.shape[0]):

    col_batch = []
    # Extract a batch of data
    for col in range(0, df_test.shape[1], 3):
        batch = df_test.iloc[row, col:col+3]
        # Rename the columns
        batch.index = ['X', 'Y', 'Z']
        #print(f'batch: {batch}')
        col_batch.append(batch)
    # Append the batch to the list
    batches.append(col_batch)

# Concatenate the batches of data under each other
concat_batches = []
for batch in batches:
    concat_batch = pd.concat(batch, axis=1).transpose()
    print('Reshaped dataframes:', concat_batch)
    concat_batches.append(concat_batch)


# Create a dictionary with the eigenvalues and eigenvectors as values
eigen_dict = {'eigenvec_1_1': [],
                'eigenvec_1_2': [],
                'eigenvec_1_3': [],
                'eigenvec_2_1': [],
                'eigenvec_2_2': [],
                'eigenvec_2_3': [],
                'eigenvec_3_1': [],
                'eigenvec_3_2': [],
                'eigenvec_3_3': [],
              'eigenval_1': [],
              'eigenval_2': [],
              'eigenval_3': []}

# Create the DataFrame
eigen_df = pd.DataFrame(eigen_dict)

# Compute the covariance matrix for each batch
for concat_batch in concat_batches:
    # Compute the covariance matrix
    cov_matrix = np.cov(concat_batch, rowvar=False)
    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Add the eigenvalues and eigenvectors to the DataFrame
    # Store eigenvector 1
    eigen_dict['eigenvec_1_1'].append(eigenvectors[0, 0])
    eigen_dict['eigenvec_1_2'].append(eigenvectors[1, 0])
    eigen_dict['eigenvec_1_3'].append(eigenvectors[2, 0])
    # Store eigenvector 2
    eigen_dict['eigenvec_2_1'].append(eigenvectors[0, 1])
    eigen_dict['eigenvec_2_2'].append(eigenvectors[1, 1])
    eigen_dict['eigenvec_2_3'].append(eigenvectors[2, 1])
    # Store eigenvector 3
    eigen_dict['eigenvec_3_1'].append(eigenvectors[0, 2])
    eigen_dict['eigenvec_3_2'].append(eigenvectors[1, 2])
    eigen_dict['eigenvec_3_3'].append(eigenvectors[2, 2])
    # Store eigenvalues
    eigen_dict['eigenval_1'].append(eigenvalues[0])
    eigen_dict['eigenval_2'].append(eigenvalues[1])
    eigen_dict['eigenval_3'].append(eigenvalues[2])

# Finally add generated features to the DataFrame
df_aggregate = pd.concat([df_aggregate, pd.DataFrame(eigen_dict)], axis=1)

print('Final df:', df_aggregate)

#NEW


#TODO write valid report function for mlp
#NEW COPY TO SERVER
# ALSO NEW save_history hist
##TODO add eval_dict to every model
'''
data_report_mlp_base_raw = Report('mlp_raw', best_model, X_test_best, y_test_best, description=['MLP base', 'cv_data', 'raw features'])
print(data_report_mlp_base_raw)
precision_recall_multiclass(best_model, X_test_best, y_test_best)
y_pred = best_model.predict(X_test_best)

'''
print(eval_dict)
#plot_cm(y_test_best, y_pred)


def boxplot_model_comparison(df_result:pd.DataFrame):
    ax = sns.boxplot(data = f2_df, linewidth=1, showfliers=False)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
    sns.set(rc = {'figure.figsize':(8,10)})
    ax.set(ylabel='F2-Score')
    ax.set_title('F2-Score Deviation of different Models')


    #Func1

    # Without training
hist = load_history('mlp_base_model_raw_cv0')
plot_learning_curves(hist, hyperparams, 'test')
eval_dict = load_eval_dict_pkl('mlp_base_model_raw_eval_dict')

def plot_eval_dict_barplot_new(data):

    # Extracting data
    train_acc = data['train']['accuracy']
    train_prec = data['train']['precision']
    test_acc = data['test']['accuracy']
    test_prec = data['test']['precision']

    # Creating subplots for accuracy and precision
    fig, axs = plt.subplots(2, 2, figsize=(15, 5))
    bar_width = 0.2

    # Plot for accuracy
    axs[0][0].bar([i+bar_width/2 for i in range(len(train_acc))], train_acc, width=bar_width, color='b', label='Train Accuracy')
    axs[0][0].bar([i-bar_width/2 for i in range(len(test_acc))], test_acc, width=bar_width, color='r', label='Test Accuracy')
    axs[0][0].set_xticks(range(len(train_acc)))
    axs[0][0].set_xticklabels([f'CV{i}' for i in range(len(train_acc))])
    axs[0][0].set_ylabel('Accuracy')
    axs[0][0].set_title('Accuracy')
    axs[0][0].legend()

    # Plot for precision
    axs[0][1].bar([i+bar_width/2 for i in range(len(train_prec))], train_prec, width=bar_width, color='b', label='Train Precision')
    axs[0][1].bar([i-bar_width/2 for i in range(len(test_prec))], test_prec, width=bar_width, color='r', label='Test Precision')
    axs[0][1].set_xticks(range(len(train_prec)))
    axs[0][1].set_xticklabels([f'CV{i}' for i in range(len(train_acc))])
    axs[0][1].set_ylabel('Precision')
    axs[0][1].set_title('Precision')
    axs[0][1].legend()

    # Plot for recall #TODO fix this
    axs[1][0].bar([i+bar_width/2 for i in range(len(train_prec))], train_prec, width=bar_width, color='b', label='Train Precision')
    axs[1][0].bar([i-bar_width/2 for i in range(len(test_prec))], test_prec, width=bar_width, color='r', label='Test Precision')
    axs[1][0].set_xticks(range(len(train_prec)))
    axs[1][0].set_xticklabels([f'CV{i}' for i in range(len(train_acc))])
    axs[1][0].set_ylabel('Recall')
    axs[1][0].set_title('Recall')
    axs[1][0].legend()

    # Plot for f1_score #TODO fix this
    axs[1][1].bar([i+bar_width/2 for i in range(len(train_prec))], train_prec, width=bar_width, color='b', label='Train Precision')
    axs[1][1].bar([i-bar_width/2 for i in range(len(test_prec))], test_prec, width=bar_width, color='r', label='Test Precision')
    axs[1][1].set_xticks(range(len(train_prec)))
    axs[1][1].set_xticklabels([f'CV{i}' for i in range(len(train_acc))])
    axs[1][1].set_ylabel('F1 Score')
    axs[1][1].set_title('F1 Score')
    axs[1][1].legend()

    plt.show()

plot_eval_dict_barplot_new(eval_dict)