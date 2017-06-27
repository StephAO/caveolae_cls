import matplotlib.pyplot as plt
import os
import pickle
import sys
from pkg_resources import resource_filename


def graph_metrics(metrics):

    epochs = range(len(metrics['validation_loss']))

    f, ((t_loss, v_loss), (t_m, v_m)) = plt.subplots(2, 2, sharex='col', sharey='row')
    m, = t_loss.plot(epochs, metrics['training_loss'], 'm', label='loss')
    t_loss.set_ylabel('loss')
    t_loss.set_title('training')

    v_loss.plot(epochs, metrics['validation_loss'], 'm', label='loss')
    v_loss.set_ylabel('loss')
    v_loss.set_title('validation')

    b, = t_m.plot(epochs, metrics['training_accuracy'], 'b', label='accuracy')
    r, = t_m.plot(epochs, metrics['training_precision'], 'r', label='precision')
    g, = t_m.plot(epochs, metrics['training_recall'], 'g', label='recall')
    c, = t_m.plot(epochs, metrics['training_f1'], 'c', label='f1')
    t_m.set_xlabel('epochs')
    t_m.set_ylim(0, 1)

    v_m.plot(epochs, metrics['validation_accuracy'], 'b', label='accuracy')
    v_m.plot(epochs, metrics['validation_precision'], 'r', label='precision')
    v_m.plot(epochs, metrics['validation_recall'], 'g', label='recall')
    v_m.plot(epochs, metrics['validation_f1'], 'c', label='f1')
    v_m.set_xlabel('epochs')
    v_m.set_ylim(0, 1)

    f.legend((m,b,r,g,c), ('loss', 'accuracy', 'precision', 'recall', 'f1'), loc=4, ncol=5, mode="expand", borderaxespad=0.)
    f.suptitle("Metrics")

    plt.show()


def main():
    dir_ = resource_filename('caveolae_cls', '/data')
    if len(sys.argv) < 2:
        print "Must input pickle filename"
        exit(1)
    fn = os.path.join(dir_, sys.argv[1])
    model_info, hyper_params, metrics = pickle.load(open(fn, "rb"))
    print hyper_params
    graph_metrics(metrics)

if __name__ == "__main__":
    main()