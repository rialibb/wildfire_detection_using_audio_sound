import torch


class PCA(torch.nn.Module):
    def __init__(self, PCA_limit=10, center=True):
        """
        PCA (Principal Component Analysis) class constructor.
        
        .. todo :: Add the Import method to build the torch.Tensor with the features and their values
                   using the class in the other files.

        Parameters
        ----------
        PCA_limit : scalar
            -- number of principal component kept at the end
        center : booleen
            -- Whether to center the feature_value. Default is True.
        """

        self.center = center
        self.PCA_limit = PCA_limit

    def SVD(self, feature_value):
        """
        Fit the PCA model to the input feature_value.
        Computes and stores the principal components and explained variance.
        
        Parameters
        ----------
        x : torch.Tensor
            -- Input feature_value tensor

        """
        if self.center:
            feature_value = feature_value - torch.mean(feature_value, dim=0)

        n_samples, n_features = feature_value.shape
        self.n_samples = n_samples

        self.PCA_limit = n_features

        U, S, V = torch.svd(feature_value)

        self.components = V.T[:, :self.PCA_limit]
        self.explained_variance = torch.mul(
            S[0:self.PCA_limit], S[0:self.PCA_limit]) / (n_samples - 1)

    def extract(self, feature_value):
        """
        Reduce the dimensionality of the input feature_value by projecting it onto the principal components.
        
        Parameters
        ----------
        feature_valuex : torch.Tensor
            -- Input feature_value tensor.
        
        Returns
        -------
         new_features : torch.Tensor
            -- Transformed feature_value using the principal components.

        """
        if self.center:
            feature_value = feature_value - torch.mean(feature_value, dim=0)
        
        self.new_features=torch.matmul(feature_value, self.components)

        return self.new_features


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
    print("Explained Variance:")
    print(pca.explained_variance)
