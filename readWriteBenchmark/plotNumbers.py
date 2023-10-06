import matplotlib.pyplot as plt
import numpy as np


def plot(kernelTime, stdError, inputSizeString):
    offsets = ['#0', '#16', '#20', '#24', '#128']
    x_pos = np.arange(len(offsets))

    # Build the plot
    fig, ax = plt.subplots()
    ax.bar(x_pos, kernelTime, yerr=stdError, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Runtime (ns)')
    ax.set_yscale('log')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(offsets)
    ax.set_title(inputSizeString + "MB Input Size")

    # Save the figure and show
    plt.tight_layout()
    plt.savefig("bar" + inputSizeString + ".png")
    plt.show()



def runPlots():
    ## 256MB
    kernelTime = np.array([1.32E+06,	1318600,	1317860,	1317450,	1321190])
    stdError =    np.array([5479.09,	4893.69,	3603.21,	3909.52,	4257.92])
    plot(kernelTime, stdError, "256")

    ## 512MB
    kernelTime =  np.array([2633610,	2624280,	2624320,	2623390,	2627670])
    stdError =    np.array([9178.74,	4716.37,	5408.13,	4899.64,	7913.9])
    plot(kernelTime, stdError, "512")

    ## 1024MB
    kernelTime =  np.array([5259590,	5240110,	5243420,	5241390,	5249470])
    stdError =    np.array([14079.5,	6567.98,	9061.81,	7142.75,	13986])
    plot(kernelTime, stdError, "1024")



if __name__ == "__main__":
    runPlots()