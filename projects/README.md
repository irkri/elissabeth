# Elissabeth Projects

Each folder here contains experiments where Elissabeth models train on given data.
For each project, several files are given:

- `data.py`, containing methods or classes for data handling and creation
- `main.py`, the script callable from a terminal, which starts training
- `config.json`, the configuration file for the current Elissabeth model
- `watcher.ipynb`, a python notebook containing tools to analyze a trained Elissabeth model

## Starting an experiment

We advise to browse through the code to get used to our pipeline. Starting an experiment is
simple. Below is a walkthrough of a typical test run.

1. Configure the model by modifying the `config.json` file.
2. Configure data and training hyperparameters, which are found at the top of each `main.py` file
   in a Python dictionary called `config`.
3. Start an experiment run by calling

        $ python main.py train
4. Test the trained model by calling

        $ python main.py test --load [your-model-id]
    where `your-model-id` is depending on your experiment tracker and logger. An experiment is
    typically saved in your current experiment folder. The model ID is the name of the folder
    containing the checkpoint file.
5. Observe model parameters and model behavior in specific test cases in the `watcher.ipynb`
   notebook, again loading the model with the given model ID.


### Configuration config.json

Below are the most important hyperparameters in the config.json file explained. These parameters
can be found with the same name in configuration classes of sainomore modules.

- *context_length* - Length $T$ of the longest possible input.,
- *input_type* - Either `"vector"` or `"token"`.
- *input_vocab_size* - Number of possible input tokens for *input_type*=`"token"`, else vector
dimensionality.
- *output_vocab_size* - Dimensions of the output of the whole Elissabeth model. If it is not
specified, it is assumed to be the same as *input_vocab_size*.
- *n_layers* - Number of layers $L$.,
- *d_hidden* - Size of the latent space throughout the model $d_{\text{hidden}}$.
- *layer_norm* - If layer norm should be used, boolean.
- *residual_stream*: If the residual stream should be used, boolean.
- *semiring* - One of `"real"`, `"arctic"` or `"bayesian"`. The semiring throughout the model.
- *n_is* - Number of iterated sums $N$ in all layers.
- *lengths* - List of lengths of the ISS that one LISS layer computes. Often this is a list with
one integer, e.g. `[3]`.
- *sum_normalization* - Whether to use sum normalization for the ISS, boolean.
- *d_values* - Dimensionality of the values in LISS $d_v$.
- *values_2D* - Whether the values are 2-dimesional, i.e. matrices instead of vectors are used,
boolean.
- *pe_value* - Whether positional encoding is used, boolean.
- *v_norm* - Whether the value vectors should be normalized after creation, boolean.
- *v_shared* - If set to true, values are shared abroad all $N$ iterated sums, boolean.
- *restrict_query_key* - If set to true, query and key are restricted in size by some activation
function that depends on the current weighting, boolean.
- *d_query_key* - Dimensionality of query and key $d_{q,k}$.
- *exp_alpha_0* - The parameter $\alpha_0$ for the exponential weighting
*weighting*=``["ExponentialDecay"]``.
- *qk_activation*: Activation function of the query-key transform. Either ``none`` for linear
transformation or ``"relu"``/``"sine"`` for an FFN.
- *qk_latent* - Latent dimension of the FFN when *qk_activation* is set to an activation function.
- *qk_include_time* - If the time component should be given to the FFN, see above, boolean.
- *v_activation* - Same as for query-key.
- *v_latent* - Same as for query-key.
- *v_include_time* - Same as for query-key.
- *weighting* - A list of weightings that can be used in an iterated sum. Possible options are
  ``"ExponentialDecay"``, ``"Exponential"``, ``"CosineDecay"``, ``"Cosine"``,
  ``"ComplexExponential"`` or ``"MSC"``.
