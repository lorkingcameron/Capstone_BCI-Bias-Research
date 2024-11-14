import matplotlib.pyplot as plt
import statistics

def generate_plot():
    random_seed = 1165134387
    accuracies = [60, 90, 60, 70, 90, 80, 70, 100, 70, 70]
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