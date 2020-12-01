from moa.feature_encoder import NumericalFeatureEncoder, EmbeddingFeatureEncoder,  CategoricalFeatureEncoder


class FeatureTransformer:
    """
    Feature encoder for different feature types
    """
    def __init__(self, gene_features=None, cell_features=None, categorical_features=None):
        self.gene_features = gene_features
        self.cell_features = cell_features
        self.categorical_features = categorical_features

    def transform(self, X):
        gene_feature_inputs, gene_feature_encoders = NumericalFeatureEncoder(self.gene_features).encode(X) 
        cell_feature_inputs, cell_feature_encoders = NumericalFeatureEncoder(self.cell_features).encode(X) 
        categorical_inputs, categorical_feature_encoders = CategoricalFeatureEncoder(self.categorical_features).encode(X)

        feature_layer_inputs = {
                                **gene_feature_inputs,
                                **cell_feature_inputs,
                                **categorical_inputs
                                }
 
        return feature_layer_inputs, categorical_feature_encoders, gene_feature_encoders, cell_feature_encoders
