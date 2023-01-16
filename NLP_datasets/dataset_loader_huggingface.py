import transformers
import torch as th
import numpy as np
import pandas as pd

from datasets import load_dataset


if __name__ == "__main__":
    dataset = load_dataset("ag_news")

    print(type(dataset))
    print(dataset["train"][0])
    print(len(dataset["train"]), len(dataset["test"]))