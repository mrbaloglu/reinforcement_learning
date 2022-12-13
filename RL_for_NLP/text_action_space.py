from typing import List
from gym.spaces.discrete import Discrete
import pandas as pd


class ActionSpace(Discrete):
    def __init__(self, actions: List[str]):
        self.actions = actions
        self._ix_to_action = {ix: action for ix, action in enumerate(self.actions)}
        self._action_to_ix = {action: ix for ix, action in enumerate(self.actions)}
        super().__init__(len(self.actions))

    def __post_init__(self):
        self._ix_to_action = {ix: action for ix, action in enumerate(self.actions)}
        self._action_to_ix = {action: ix for ix, action in enumerate(self.actions)}

    def action_to_ix(self, action: str) -> int:
        return self._action_to_ix[action]

    def ix_to_action(self, ix: int) -> str:
        return self._ix_to_action[ix]

    def size(self) -> int:
        return self.n

    def __repr__(self):
        return f"Discrete Action Space with {self.size()} actions: {self.actions}"
    
    def __len__(self):
        return len(self.actions)

if __name__ == "__main__":
    data = pd.read_csv("NLP_datasets/rt-polarity-full.csv")
    data.columns = ['label', 'review']
    data["label_str"] = data["label"].apply(lambda x: "good" if x == 1 else "bad")
    a_s = ActionSpace(list(data["label_str"].unique()))
    print(a_s)