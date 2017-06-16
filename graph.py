import matplotlib.pyplot as plt
import os
import pickle
import sys
from pkg_resources import resource_filename


def graph_metrics(metrics):

    epochs = range(len(metrics['v_loss']))

    plt.figure(1)
    plt.subplot(211)
    plt.plot(epochs, metrics['v_loss'], 'b', label='validation')
    plt.plot(epochs, metrics['t_loss'], 'r', label='training')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.ylim(0, 1)
    plt.ylabel('loss')

    plt.subplot(212)
    plt.plot(epochs, metrics['v_acc'], 'b', label='validation')
    plt.plot(epochs, metrics['t_acc'], 'r', label='training')
    plt.ylabel('accuracy (%)')
    plt.ylim(0, 1)
    plt.xlabel('epochs')

    plt.show()


def main():
    dir_ = resource_filename('caveolae_cls', '/data')
    if len(sys.argv) < 2:
        print "Must input pickle filename"
        exit(1)
    fn = os.path.join(dir_, sys.argv[1])
    model_info, hyper_params, metrics = pickle.load(open(fn, "rb"))
    graph_metrics(metrics)

if __name__ == "__main__":
    main()