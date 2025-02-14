from pathlib import Path
from typing import Tuple, Union
import h5py
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input
from tensorflow.python.keras.saving import hdf5_format

from ms2deepscore import SpectrumBinner


class SiameseModel:
    """
    Class for training and evaluating a siamese neural network, implemented in Tensorflow Keras.
    It consists of a dense 'base' network that produces an embedding for each of the 2 inputs. The
    'head' model computes the cosine similarity between the embeddings.

    Mimics keras.Model API.

    For example:

    .. code-block:: python

        # Import data and reference scores --> spectrums & tanimoto_scores_df

        # Create binned spectrums
        spectrum_binner = SpectrumBinner(1000, mz_min=10.0, mz_max=1000.0, peak_scaling=0.5)
        binned_spectrums = spectrum_binner.fit_transform(spectrums)

        # Create generator
        dimension = len(spectrum_binner.known_bins)
        test_generator = DataGeneratorAllSpectrums(binned_spectrums, tanimoto_scores_df,
                                                   dim=dimension)

        # Create (and train) a Siamese model
        model = SiameseModel(spectrum_binner, base_dims=(600, 500, 400), embedding_dim=400,
                             dropout_rate=0.2)
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.001))
        model.summary()
        model.fit(test_generator,
                  validation_data=test_generator,
                  epochs=50)

    """
    def __init__(self,
                 spectrum_binner: SpectrumBinner,
                 base_dims: Tuple[int, ...] = (600, 500, 500),
                 embedding_dim: int = 400,
                 dropout_rate: float = 0.5,
                 dropout_in_first_layer: bool = False,
                 l1_reg: float = 1e-6,
                 l2_reg: float = 1e-6,
                 keras_model: keras.Model = None):
        """
        Construct SiameseModel

        Parameters
        ----------
        spectrum_binner
            SpectrumBinner which is used to bin the spectra data for the model training.
        base_dims
            Tuple of integers depicting the dimensions of the desired hidden
            layers of the base model
        embedding_dim
            Dimension of the embedding (i.e. the output of the base model)
        dropout_rate
            Dropout rate to be used in the base model.
        dropout_in_first_layer
            Set to True if dropout should be part of first dense layer as well. Default is False.
        l1_reg
            L1 regularization rate. Default is 1e-6.
        l2_reg
            L2 regularization rate. Default is 1e-6.
        keras_model
            When provided, this keras model will be used to construct the SiameseModel instance.
            Default is None.
        """
        # pylint: disable=too-many-arguments
        assert spectrum_binner.known_bins is not None, \
            "spectrum_binner does not contain known bins (run .fit_transform() on training data first!)"
        self.spectrum_binner = spectrum_binner
        self.input_dim = len(spectrum_binner.known_bins)

        if keras_model is None:
            # Create base model
            self.base = self.get_base_model(input_dim=self.input_dim,
                                            base_dims=base_dims,
                                            embedding_dim=embedding_dim,
                                            dropout_rate=dropout_rate,
                                            dropout_in_first_layer=dropout_in_first_layer,
                                            l1_reg=l1_reg,
                                            l2_reg=l2_reg)
            # Create head model
            self.model = self._get_head_model(input_dim=self.input_dim,
                                              base_model=self.base)
        else:
            self._construct_from_keras_model(keras_model)

    def save(self, filename: Union[str, Path]):
        """
        Save model to file.

        Parameters
        ----------
        filename
            Filename to specify where to store the model.

        """
        with h5py.File(filename, mode='w') as f:
            hdf5_format.save_model_to_hdf5(self.model, f)
            f.attrs['spectrum_binner'] = self.spectrum_binner.to_json()

    @staticmethod
    def get_base_model(input_dim: int,
                       base_dims: Tuple[int, ...] = (600, 500, 500),
                       embedding_dim: int = 400,
                       dropout_rate: float = 0.25,
                       dropout_in_first_layer: bool = False,
                       l1_reg: float = 1e-6,
                       l2_reg: float = 1e-6,
                       dropout_always_on: bool = False) -> keras.Model:
        """Create base model for Siamaese network.

        Parameters
        ----------
        input_dim : int
            Dimension of the input vectors.
        base_dims
            Tuple of integers depicting the dimensions of the desired hidden
            layers of the base model
        embedding_dim
            Dimension of the embedding (i.e. the output of the base model)
        dropout_rate
            Dropout rate to be used in the base model
        dropout_in_first_layer
            Set to True if dropout should be part of first dense layer as well. Default is False.
        l1_reg
            L1 regularization rate. Default is 1e-6.
        l2_reg
            L2 regularization rate. Default is 1e-6.
        dropout_always_on
            Default is False in which case dropout layers will only be active during
            model training, but switched off during inference. When set to True,
            dropout layers will always be on, which is used for ensembling via
            Monte Carlo dropout.
        """
        # pylint: disable=too-many-arguments
        
        dropout_starting_layer = 0 if dropout_in_first_layer else 1
        model_input = Input(shape=input_dim, name='base_input')
        for i, dim in enumerate(base_dims):
            if i == 0: # L1 and L2 regularization only in 1st layer
                model_layer = Dense(dim, activation='relu', name='dense'+str(i+1),
                                    kernel_regularizer=keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg))(
                   model_input)
            else:
                model_layer = Dense(dim, activation='relu', name='dense'+str(i+1))(model_layer)

            model_layer = BatchNormalization(name='normalization'+str(i+1))(model_layer)
            if dropout_always_on and i >= dropout_starting_layer:
                model_layer = Dropout(dropout_rate, name='dropout'+str(i+1))(model_layer,
                                                                             training=True)
            elif i >= dropout_starting_layer:
                model_layer = Dropout(dropout_rate, name='dropout'+str(i+1))(model_layer)

        embedding = Dense(embedding_dim, activation='relu', name='embedding')(model_layer)
        return keras.Model(model_input, embedding, name='base')

    @staticmethod
    def _get_head_model(input_dim: int,
                        base_model: keras.Model):
        input_a = Input(shape=input_dim, name="input_a")
        input_b = Input(shape=input_dim, name="input_b")
        embedding_a = base_model(input_a)
        embedding_b = base_model(input_b)
        cosine_similarity = keras.layers.Dot(axes=(1, 1),
                                             normalize=True,
                                             name="cosine_similarity")([embedding_a, embedding_b])
        return keras.Model(inputs=[input_a, input_b], outputs=[cosine_similarity],
                           name='head')

    def _construct_from_keras_model(self, keras_model):
        def valid_keras_model(given_model):
            assert given_model.layers, "Expected valid keras model as input."
            assert len(given_model.layers) > 2, "Expected more layers"
            assert len(keras_model.layers[2].layers) > 1, "Expected more layers for base model"
            
        valid_keras_model(keras_model)
        self.base = keras_model.layers[2]
        self.model = keras_model

    def compile(self, *args, **kwargs):
        self.model.compile(*args, **kwargs)

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)

    def load_weights(self, checkpoint_path):
        self.model.load_weights(checkpoint_path)

    def summary(self):
        self.base.summary()
        self.model.summary()

    def evaluate(self, *args, **kwargs):
        return self.model.evaluate(*args, **kwargs)
