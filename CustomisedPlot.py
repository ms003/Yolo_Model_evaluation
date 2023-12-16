import matplotlib.pyplot as plt
import numpy as np
import os



class CustomisedPlot:

    def __init__(self):
        pass

    def plot_scatter_2d(self, X,Y, title, X_label, Y_label, save_png_path, x_lim, y_lim, optional_annotation = ''):
        #fig = plt.figure()
        #ax = fig.add_axes([0, 0, 1, 1])

        fig, ax = plt.subplots(figsize=(10, 10.8))
        ax.scatter(X, Y, color='r')
        ax.set_xlabel(X_label, fontsize = 18)
        ax.set_ylabel(Y_label, fontsize = 18)
        ax.set_title(title)

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        plt.grid()

        # annotation at each data point
        for x,y, annotation_str in zip(X,Y, optional_annotation):
            ax.annotate(annotation_str, (x, y), fontSize=20)

        save_png_filename = save_png_path + '/' + title + '.png'
        print(save_png_filename)
        fig.savefig(save_png_filename)

        return save_png_filename

    def init_plot_scatter_2d(self,  X_label, Y_label, title, x_lim, y_lim):

        # fig = plt.figure()
        # ax = fig.add_axes([0, 0, 1, 1])

        fig, ax = plt.subplots(figsize=(10, 10))

        ax.set_xlabel(X_label, fontsize = 18)
        ax.set_ylabel(Y_label, fontsize = 18)
        ax.set_title(title, fontsize = 18)

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        plt.grid()

        return fig, ax

    def plot_scatter_2d_without_save(self, fig, ax, X, Y, best_mAP, best_mAR, median_mAP, median_mAR):
        X = np.array(X)

        Y = np.array(Y)

        ax.scatter(X, Y, marker='o', s=15)
        ax.scatter(best_mAR, best_mAP, marker='o', s=50, label='best_mAP/mAR')
        ax.scatter(median_mAR, median_mAP, marker='o', s=50, label='median_mAP/mAR')



        ax.legend()

        plt.show()

        #fig.savefig(save_png_path + '/' + 'precision_vs_accuracy' + '.png')

        return fig, ax

    def plot_f1_score(self, list_mF1_all):
        list_mF1_all =np.array(list_mF1_all)

        fig, ax = plt.subplots(figsize=(10, 10))

        ax.set_xlabel('F1_score', fontsize = 18)
        ax.set_ylabel('Counts', fontsize = 18)
        ax.set_title('F1_score frequency', fontsize = 18)

        ax.set_xlim(min(list_mF1_all)-0.05, 1)


        plt.grid(axis ='y')

        ax.hist(list_mF1_all, histtype ='step', color = 'magenta', alpha= 0.5 )  # label=optional_annotation, c=np.random.rand(3,)

        plt.legend()

        #plt.savefig(save_png_path + '/' + 'F1_score_frequency' + '.png')
        plt.show()

        return fig, ax

    def save_fig(self,  fig, ax, save_png_path, title):

        save_png_filename = save_png_path + '/' + title + '.png'
        fig.savefig(save_png_filename, dpi=150)

        return save_png_filename
