"""claimbreak: when do label-based spatial conclusions break?

Start from a real tissue whose labels are taken as the reference, optionally plant known signals, introduce
label and count errors in a controlled way, and record which conclusions survive.
"""
from .tissue import Tissue, load, split, thin, describe_file
from .labeling import NearestCentroid, native_confusion, recall_labels
from . import plant, perturb, questions, compare, flipmodel, stress, compartment, geometry, exposure
from .experiment import run, load_config

__version__ = "1.7.103"
__all__ = ["Tissue", "load", "split", "thin", "describe_file", "NearestCentroid", "native_confusion",
           "recall_labels", "plant", "perturb", "questions", "compare", "flipmodel", "stress", "compartment", "geometry", "exposure", "run",
           "load_config"]
