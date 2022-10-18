# A GPU acceleration plugin for scikit-learn

> This is a proof-of-concept. Everything might change.

## Install

To install this you need to install a custom verison of scikit-learn as well
as this plugin.

1. Create a new conda environment with `mamba create -n sklearn-gpu -c rapidsai -c nvidia python scikit-learn cuml ipython compilers cython`

2. Checkout the code for [pull request #24497](https://github.com/scikit-learn/scikit-learn/pull/24497)
   and install it with `pip install --no-build-isolation -e .`.

3. Install this plugin by checking out the code and `pip install -e .`.


## Running

To try it out enable the plugin using:

```python
with sklearn.config_context(engine_provider="sklearn_gpu"):
    km = KMeans()
    km.fit(X)
    y_pred = km.predict(X)
```