import numpy as np
from typing import List
from dataclasses import dataclass


@dataclass(init=True)
class Observation:
    sample_str: str
    sample_vecs: List[np.ndarray]
    label_str: str
    label_enc: int

    def get_sample_str(self) -> str:
        return self.sample_str

    def get_sample_vecs(self) -> List[np.ndarray]:
        return self.sample_vecs

    def get_label_str(self) -> str:
        return self.label_str
    
    def get_label_enc(self) -> int:
        return self.label_enc

    def __str__(self) -> str:
        return f"Text: {self.sample_str}, Label: {self.label_str}"

@dataclass(init=True)
class BertObservation:
    sample_str: str
    sample_input_id_vecs: List[np.ndarray]
    sample_token_type_vecs: List[np.ndarray]
    sample_attn_mask_vecs: List[np.ndarray]
    label_str: str
    label_enc: int

    def get_sample_str(self) -> str:
        return self.sample_str

    def get_sample_input_id_vecs(self) -> List[np.ndarray]:
        return self.sample_input_id_vecs
    
    def get_sample_token_type_vecs(self) -> List[np.ndarray]:
        return self.sample_token_type_vecs
    
    def get_sample_attn_mask_vecs(self) -> List[np.ndarray]:
        return self.sample_attn_mask_vecs

    def get_label_str(self) -> str:
        return self.label_str
    
    def get_label_enc(self) -> int:
        return self.label_enc

    def __str__(self) -> str:
        return f"Text: {self.sample_str}, Label: {self.label_str}"

if __name__ == "__main__":
    obs = BertObservation()