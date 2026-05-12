import sys
import numpy as np
import pandas as pd

from pathlib import Path
ROOT_PATH = Path(__file__).resolve().parent.parent
MOVES_DS_PATH = ROOT_PATH / "datasets" / "moves.csv"

sys.path.append(str(ROOT_PATH))
from token_maps.type_token_map import type_token_map

class MovesPretrainer:
    def __init__(self):
        self.moves_df = pd.read_csv(MOVES_DS_PATH)
        self.moves_count = len(self.moves_df)

    def get_stats(self, move_token:int):
        # move_token is pokemon_id + 1 (tokens 0 and 1 are reserved for null and unk)
        # predicted_stats in the order [type, power, PP, accuracy, min hits, max hits, is_special, is_physical, is_status]

        move = self.moves_df.iloc[move_token - 2]
        
        true_stats = [
            type_token_map.get(move["type"].lower(), 1),
            move["power"],
            move["pp"],
            move["accuracy"],
            move["min_hits"],
            move["max_hits"],
            1 if move["damage_class"] == "Special" else 0,
            1 if move["damage_class"] == "Physical" else 0,
            1 if move["damage_class"] == "Status" else 0
        ]

        return true_stats
    
    def get_shuffled_batch(self):
        tokens = np.arange(2, self.moves_count + 2)
        np.random.shuffle(tokens)

        batch_list = []
        for token in tokens:
            row = [token] + self.get_stats(token)
            batch_list.append(row)

        batch = np.asarray(batch_list)

        return batch