import tensorflow as tf

from moa.base_classifier import BaseClassifier
from moa.transformer import FeatureTransformer


class MOAClassifier(BaseClassifier):
    """
    Setup network architecture
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compile_model(self, x_train=None, y_train=None):
        data_types = x_train.dtypes
        categorical_features = ['cp_type', 'cp_dose', 'cp_time']
        numerical_features = data_types[data_types=='float64'].index.tolist()
        numerical_features_gene = [feature for feature in numerical_features if 'g' in feature]
        numerical_features_cell = [feature for feature in numerical_features if 'c' in feature]

        total_classes = y_train.shape[-1]
        inputs, linear_feature_columns, gene_feature_columns, cell_feature_columns = FeatureTransformer(gene_features=numerical_features_gene,
                                                                                                        cell_features=numerical_features_cell,
                                                                                                        categorical_features=categorical_features)\
                                                                                     .transform(x_train)

        metrics = [tf.keras.metrics.BinaryAccuracy(name='accuracy'), tf.keras.metrics.AUC(name='auc')]

        gene_deep = tf.keras.layers.DenseFeatures(gene_feature_columns)(inputs)
        cell_deep = tf.keras.layers.DenseFeatures(cell_feature_columns)(inputs)
        wide = tf.keras.layers.DenseFeatures(linear_feature_columns)(inputs)

        for numnodes in [128]:
            gene_deep = tf.keras.layers.Dense(numnodes, activation='relu')(gene_deep)
            gene_deep = tf.keras.layers.Dropout(0.2)(gene_deep)

        for numnodes in [128]:
            cell_deep = tf.keras.layers.Dense(numnodes, activation='relu')(cell_deep)
            cell_deep = tf.keras.layers.Dropout(0.2)(cell_deep)

        deep = tf.keras.layers.concatenate([gene_deep, cell_deep])

        for numnodes in [64, 32]:
            deep = tf.keras.layers.Dense(numnodes, activation='relu')(deep)
            deep = tf.keras.layers.Dropout(0.2)(deep)

        combined = tf.keras.layers.concatenate([deep, wide])
        output = tf.keras.layers.Dense(total_classes, activation='sigmoid')(combined)
        model = tf.keras.Model(inputs=[v for v in inputs.values()], outputs=output)
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.0),
                      optimizer=tf.keras.optimizers.Adam(lr=0.0005),
                      metrics=metrics)
        return model
