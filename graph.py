import matplotlib.pyplot as plt
import pickle


def graph(title, y, y_axis='loss', x=None, x_axis='epoch'):
    if x is None:
        x = range(len(y))
    plt.plot(x, y)
    plt.ylabel(y_axis)
    plt.xlabel(x_axis)
    plt.show()


def main():
    model_info, hyper_params, metrics = pickle.load(open('save.p', "rb"))
    for key, value in metrics:
        graph(key, value, y_axis=key.split('_')[1])

if __name__ == "__main__":
    main()