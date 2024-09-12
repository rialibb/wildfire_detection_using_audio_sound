import torch


class PCA(torch.nn.Module):
    def __init__(self, PCA_limit : int = 10, center: bool = True):
        """
        PCA (Principal Component Analysis) class constructor.
        
        .. todo :: Add the Import method to build the torch.Tensor with the features and their values
                   using the class in the other files.

        Parameters
        ----------
        PCA_limit : int
            -- number of principal component kept at the end
        center : bool
            -- Whether to center the feature_value. Default is True.
        """

        super(PCA, self).__init__()
        self.center         = center
        self.PCA_limit      = PCA_limit

        self.n_samples      = None
        self.components     = None
        self.explained_variance = None

    def SVD(self, feature_value):
        """
        Fit the PCA model to the input feature_value.
        Computes and stores the principal components and explained variance.
        
        Parameters
        ----------
        feature_value : torch.Tensor
            -- Input feature_value tensor

        """
        # center the values
        if self.center:
            feature_value = feature_value - torch.mean(feature_value, dim=0)

        # retrieve shapes
        self.n_samples = feature_value.shape[0]
        self.PCA_limit = feature_value.shape[1]

        # perform PCA
        U, S, V = torch.svd(feature_value)

        self.components = V.T[:, :self.PCA_limit]
        self.explained_variance = torch.mul(
            S[0:self.PCA_limit], S[0:self.PCA_limit]) / (n_samples - 1)

    def extract(self, feature_value):
        """
        Reduce the dimensionality of the input feature_value by projecting it onto the principal components.
        
        Parameters
        ----------
        feature_value : torch.Tensor
            -- Input feature_value tensor.
        
        Returns
        -------
         new_features : torch.Tensor
            -- Transformed feature_value using the principal components.

        """

        if self.center:
            # center values
            feature_value = feature_value - torch.mean(feature_value, dim=0)
        # compute new features
        new_features = torch.matmul(feature_value, self.components)
        return new_features


# Example of usage:
if __name__ == "__main__":
    # Generate fictitious data for the example
    n_samples = 100
    n_features = 5
    X = torch.randn(n_samples, n_features)

    pca = PCA(PCA_limit=2)

    # Fit the data
    pca.SVD(X)

    # Extract data using the top principal components
    transformed_data = pca.extract(X)

    print("Top Principal Components:")
    print(pca.components)
    print(f"Explained Variance: {pca.explained_variance}")