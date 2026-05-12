import sys
import numpy as np
import jax.numpy as jnp
import pandas as pd
import itertools

from pathlib import Path
ROOT_PATH = Path(__file__).resolve().parent.parent
TYPES_DS_PATH = ROOT_PATH / "datasets" / "types.csv"

sys.path.append(str(ROOT_PATH))
from token_maps.type_token_map import type_token_map

class TypesPretrainerE1:
    def __init__(self):
        self.types_df = pd.read_csv(TYPES_DS_PATH)
        self.types_count = 18

    def get_eff(self, type_token:int):
        # type_token is type_id + 1 (tokens 0 and 1 are reserved for null and unk)
        # predicted_eff in the order [normal, fire, water, electric, grass, ice, fighting, poison, ground, flying, psychic, bug, rock, ghost, dragon, dark, steel, fairy]

        type = self.types_df.iloc[type_token - 2]

        true_eff = [
            type["normal"],
            type["fire"],
            type["water"],
            type["electric"],
            type["grass"],
            type["ice"],
            type["fighting"],
            type["poison"],
            type["ground"],
            type["flying"],
            type["psychic"],
            type["bug"],
            type["rock"],
            type["ghost"],
            type["dragon"],
            type["dark"],
            type["steel"],
            type["fairy"]
        ]

        return true_eff
    
    def get_shuffled_batch(self):
        tokens = np.arange(2, self.types_count + 2)
        np.random.shuffle(tokens)

        batch_list = []
        for token in tokens:
            row = [token] + self.get_eff(token)
            batch_list.append(row)

        batch = jnp.array(batch_list)

        return batch

class TypesPretrainerE2:
    def __init__(self):
        self.e1 = TypesPretrainerE1()

    def get_eff(self, move_type_token:int, opp_type1_token:int, opp_type2_token:int):
        if move_type_token < 2: raise Exception("move_type_token cannot be null or unk")
        elif opp_type1_token < 2: raise Exception("opp_type1_token cannot be null or unk")

        effectivenesses = self.e1.get_eff(move_type_token)
        eff1 = effectivenesses[opp_type1_token - 2]
        eff2 = effectivenesses[opp_type2_token - 2] if opp_type2_token >= 2 else 1

        eff = eff1 * eff2

        return eff

    def get_shuffled_batch(self):
        type_tokens = np.arange(2, self.e1.types_count + 2)
        type2_tokens = np.concatenate(([0], type_tokens))

        combinations = list(itertools.product(type_tokens, type_tokens, type2_tokens))
        combinations = np.array(combinations)
        np.random.shuffle(combinations)

        batch_list = []
        for m_t, ot1, ot2 in combinations:
            eff = self.get_eff(m_t, ot1, ot2)
            batch_list.append([m_t, ot1, ot2, eff])

        return jnp.array(batch_list)
