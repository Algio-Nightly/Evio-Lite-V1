import sys
import numpy as np
import pandas as pd

from pathlib import Path
ROOT_PATH = Path(__file__).resolve().parent.parent
SPECIES_DS_PATH = ROOT_PATH / "datasets" / "species.csv"

sys.path.append(str(ROOT_PATH))
from token_maps.type_token_map import type_token_map

class SpeciesPretrainer:
    def __init__(self):
        self.species_df = pd.read_csv(SPECIES_DS_PATH)
        self.species_count = len(self.species_df)

    def get_stats(self, species_token:int):
        # species_token is pokemon_id + 1 (tokens 0 and 1 are reserved for null and unk)
        # predicted_stats in the order [type-1, type-2, hp, atk, def, sp. atk, sp. def, spd]

        pokemon = self.species_df.iloc[species_token - 2]

        types = pokemon["types"].lower().split(",")
        if len(types) < 2: types.append("<null>")
        types[0] = type_token_map.get(types[0], 1)
        types[1] = type_token_map.get(types[1], 1)

        true_stats = [
            types[0],
            types[1],
            pokemon["hp"],
            pokemon["attack"],
            pokemon["defense"],
            pokemon["special_attack"],
            pokemon["special_defense"],
            pokemon["speed"]
        ]

        return true_stats
    
    def get_shuffled_batch(self):
        tokens = np.arange(2, self.species_count + 2)
        np.random.shuffle(tokens)

        batch_list = []
        for token in tokens:
            row = [token] + self.get_stats(token)
            batch_list.append(row)

        batch = np.asarray(batch_list)

        return batch
