import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-file', default='../logs/genetic_search_supernet.log', help='log file')
    parser.add_argument('--mode', type=str, default='genetic_search', help="the mode of plotting. "
                        "['accuracy', 'random_search', 'genetic_search']")
    parser.add_argument('--title', type=str, default='Genetic Search Supernet', help='Title of the plot')
    parser.add_argument('--save-dir', type=str, default='./images', help='save dir name')
    parser.add_argument('--save-file', type=str, default='genetic_search_supernet.png', help='save file name')
    parser.add_argument('--old-ver', action='store_true', help='Whether the log is old version.')
    args = vars(parser.parse_args())
    return args


def plot(rows, iter):
    if 'accuracy' in args['mode']:
        train_acc_list = []
        val_acc_list = []

        for row in rows:
            if row[0] != '[':
                continue
            elif row[:len('[Trainer]')] == '[Trainer]' or \
                row[:len('[Maintainer]')] == '[Maintainer]':
                continue

            epoch = int(row[row.find(' ') + 1: row.find(']')])

            if 'training' in row:
                train_acc = float(row[row.find('=') + 1:])
                train_acc_list.append((epoch, train_acc))

            if 'validation' in row:
                val_acc = float(row[row.find('=') + 1: row.find('=') + 9])
                val_acc_list.append((epoch, 1 - val_acc))

        # plot the accuracies
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, len(train_acc_list)), [item[1] for item in train_acc_list], label="train_top1")
        plt.plot(np.arange(0, len(val_acc_list)), [item[1] for item in val_acc_list], label="val_top1")
        plt.title(args["title"])
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(args["save_dir"], args["save_file"]))
        plt.show()
    elif 'search' in args['mode']:
        score_list = []
        val_acc_list = []

        for i, row in enumerate(rows):
            if 'score' in row:
                score = float(row[row.find(':') + 1: row.rfind('.')])
                if args["old_ver"] and 'Val' not in rows[i + 1]:
                    continue
                if not args["old_ver"] and 'Val' not in rows[i - 1]:
                    continue
                score_list.append(1 - score)
            if 'Val' in row:
                val_acc = float(row[row.find(':') + 1: row.find('\n')])
                if val_acc < 0.1:
                    score_list.pop()
                    continue
                val_acc_list.append(val_acc)

        plt.style.use("ggplot")
        plt.figure()
        axes = plt.gca()
        axes.set_xlim([-0.005, 0.5])
        axes.set_ylim([0.15, 0.7])
        summed_scores = [score_list[i] + val_acc_list[i] for i in range(len(val_acc_list))]
        color_list = ['steelblue' if summed_score != max(summed_scores) else 'indianred'
                      for summed_score in summed_scores]
        plt.scatter(score_list, val_acc_list, alpha=0.8, c=color_list, s=50, label='subnet')
        plt.title(args["title"] + ' iter' + str(iter))
        plt.xlabel("Normalized score difference (larger is better)")
        plt.ylabel("Accuracy")
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(args["save_dir"], args["save_file"][:-4] + '_iter' + str(iter) + '.png'))
        # plt.show()
        plt.close()


def parse_iter(rows):
    indices = [[-1]]
    for i, row in enumerate(rows):
        if 'Searching iter' in row:
            indices[-1].append(i - 1)
            indices.append([i])
    indices[-1].append(len(rows))
    return indices[1:]


if __name__ == '__main__':
    args = parse_args()
    rows = open(args['log_file']).readlines()
    if not os.path.exists(args['save_dir']):
        os.makedirs(args['save_dir'])

    if 'accuracy' in args['mode'] or 'random' in args['mode']:
        plot(rows, iter=0)
    else:
        iter_indices = parse_iter(rows)
        for i, iter_index in enumerate(iter_indices):
            plot(rows[iter_index[0]: iter_index[1]], i)
