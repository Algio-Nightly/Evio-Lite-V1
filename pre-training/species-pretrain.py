import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error

ROOT_PATH = Path(__file__).resolve().parent.parent
SPECIES_DS_PATH = ROOT_PATH / "datasets" / "species.csv"

class SpeciesPretrainer:
    def __init__(self):
        self.species_df = pd.read_csv(SPECIES_DS_PATH)
        self.species_count = len(self.species_df)

    def get_loss(self, species_token:int, predicted_stats:list[int]):
        # species_token is pokemon_id + 1 (tokens 0 and 1 are reserved for null and unk)
        # predicted_stats in the order [hp, atk, def, sp. atk, sp. def, spd]

        pokemon = self.species_df.iloc[species_token - 2]
        true_stats = [
            pokemon["hp"],
            pokemon["attack"],
            pokemon["defense"],
            pokemon["special_attack"],
            pokemon["special_defense"],
            pokemon["speed"]
        ]

        mse = mean_squared_error(true_stats, predicted_stats)

        return mse
    
    def get_shuffled_batch(self):
        tokens = np.arange(2, self.species_count + 1)
        np.random.shuffle(tokens)

        return tokens