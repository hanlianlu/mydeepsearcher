import os

# ignore the warnings
# None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
