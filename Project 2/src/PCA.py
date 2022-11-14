from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


class pca_extended(PCA):

    def __init__(self, n_components=None, whiten=False, svd_solver='auto'):
        super().__init__(n_components=n_components, whiten=whiten, svd_solver=svd_solver)
        self.cumulative_variance = None

    def plot_explained_variance(self):
        """ Plot of percentage of variance explained
        by the desired components"""

        self.cumulative_variance = np.cumsum(self.explained_variance_ratio_)

        plt.figure()
        plt.title('Variance over components')
        plt.plot(np.insert(self.cumulative_variance, 0, 0), 'b-*')
        plt.xlabel('Number of Components')
        plt.ylabel('Explained Variance')
        plt.hlines(y=0.9, xmin=0, xmax=self.n_components, colors='r')
        plt.legend(['PCA', '90% Threshold'])
        plt.show()

        return


class PCA_VIDEO(pca_extended):

    def __init__(self, frame_dims, n_components=None, whiten=False, svd_solver='auto'):
        super().__init__(n_components, whiten, svd_solver)

        self.frame_dims = frame_dims
        self.colors = True if len(frame_dims) == 3 else False

    def plot_mean(self):

        """ Plot of the mean value of PCA either in grayscale or
        colored depending on provided frame dimensions"""

        plt.figure()

        mean_shape = self.mean_
        mean_r = np.reshape(mean_shape, newshape=self.frame_dims)

        if self.colors:
            plt.imshow(mean_r)
        else:
            plt.imshow(mean_r, cmap='gray')

        plt.title('Mean')
        plt.show()

        return

    def show_principal_components(self, n=3):

        """! Plot of n principal components
            @param n : number of components to show
            """

        for i in range(n):
            plt.figure()
            phi = self.components_[i]
            phi_r = np.reshape(phi, newshape=self.frame_dims)

            if self.colors:
                plt.imshow(phi_r)
            else:
                plt.imshow(phi_r, cmap='gray')

            plt.title(f'Principal Component n. {i + 1}')
            plt.show()

        return

    def show_frame_projections(self, frame, n=3):

        """! Plot of projection of single frame on n principal components

            @param
            frame : specific frame to project
            n : number of components

            @return projected frame
            """

        reduced_frame = self.transform(frame[np.newaxis, :])
        projection = np.dot(reduced_frame[0, :n], self.components_[:n]).reshape(480, 640)

        plt.figure()
        plt.title(f'Frame projected on first {n} principal components')
        if self.colors:
            plt.imshow(projection)
        else:
            plt.imshow(projection, cmap='gray')
        plt.show()

        return reduced_frame
