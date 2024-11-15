import matplotlib.pyplot as plt
import statistics

def generate_custom_cnn_plot():
    random_seed = 1612385895
    accuracies = [70, 65, 45, 70, 40, 55, 80, 60, 80, 30]
    folds = [(x + 1) for x in range(len(accuracies))]
    mean = statistics.fmean(accuracies)
    plt.plot(folds, accuracies, marker='.')
    plt.suptitle('CNN Model 5 Classes Standardized - Accuracy over Folds')
    plt.title(f'Random Seed: {random_seed}', fontsize=6)
    plt.ylim(0, 100)
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Folds')
    plt.xticks(range(min(folds), max(folds)+1))
    #add horizontal average line 
    plt.axhline(
        y=mean, 
        color = 'g',
        linestyle = '--',
        label=f"Mean: {mean}%"
    )
    plt.legend(loc='lower right')
    plt.show()
    # Make sure to close the plt object once done
    plt.close()
    
    
def plot_cnn(accuracies, random_seed):
    folds = [(x + 1) for x in range(len(accuracies))]
    mean = statistics.fmean(accuracies)
    plt.plot(folds, accuracies, marker='.')
    plt.suptitle('CNN Model Accuracy over Folds')
    plt.title(f'Random Seed: {random_seed}', fontsize=6)
    plt.ylim(0, 100)
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Folds')
    plt.xticks(range(min(folds), max(folds)+1))
    #add horizontal average line 
    plt.axhline(
        y=mean, 
        color = 'g',
        linestyle = '--',
        label=f"Mean: {mean}%"
    )
    plt.legend(loc='lower right')
    plt.show()
    # Make sure to close the plt object once done
    plt.close()