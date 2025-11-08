# import all necessary libraries
import matplotlib.pyplot as plt

# function to plot losses and f1 scores of training and validation phase
def plot_losses_f1(model_names, optimal_configurations):
    # as many rows as models and 2 columns for loss and f1
    fig, axes = plt.subplots(len(model_names), 2, figsize=(12, 4 * len(model_names)))

    # loop over the models
    for i, model_name in enumerate(model_names):

        # get the data for this specific model
        config = optimal_configurations[model_name]
        epochs = list(range(1, len(config['train_losses']) + 1))

        # create the loss subplot
        ax_loss = axes[i, 0]
        ax_loss.plot(epochs, config['train_losses'], marker='o', label='Train Loss')
        ax_loss.plot(epochs, config['val_losses'], marker='o', label='Validation Loss')
        ax_loss.set_title(f"{model_name} - Loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend()
        ax_loss.grid(True)

        # create the F1 subplot
        ax_f1 = axes[i, 1]
        ax_f1.plot(epochs, config['f1_scores_train'], marker='o', label='Train F1')
        ax_f1.plot(epochs, config['f1_scores_val'], marker='o', label='Validation F1')
        ax_f1.set_title(f"{model_name} - F1 Score")
        ax_f1.set_xlabel("Epoch")
        ax_f1.set_ylabel("F1 Score")
        ax_f1.legend()
        ax_f1.grid(True)

    plt.tight_layout()
    plt.show()