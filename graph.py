import matplotlib.pyplot as plt
import pickle


def graph_metrics(metrics):

    epochs = range(len(metrics['e_loss']))

    plt.figure(1)
    plt.subplot(211)
    plt.plot(epochs, metrics['e_loss'], 'b', label='validation')
    plt.plot(epochs, [None] + metrics['t_loss'], 'r', label='training')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.ylabel('loss')

    plt.subplot(212)
    plt.plot(epochs, metrics['e_acc'], 'b', epochs, [None] + metrics['t_acc'], 'r')
    plt.ylabel('accuracy (%)')
    plt.ylim(0, 1)
    plt.xlabel('epochs')

    plt.show()


def main():
    model_info, hyper_params, metrics = pickle.load(open('save.p', "rb"))
    graph_metrics(metrics)

if __name__ == "__main__":
    main()