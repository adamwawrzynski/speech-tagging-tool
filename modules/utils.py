import matplotlib.pyplot as plt

def plot_spectrogram(spectrogram, title, x_label, y_label, figsize=(12, 5)):
    ''' Plot spectrogram '''
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.imshow(spectrogram.T, cmap=plt.cm.jet, aspect='auto')
    plt.colorbar()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    ax = plt.gca()
    ax.invert_yaxis()