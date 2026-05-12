# Pre-Training Plan (Approach 1)

- We pre-train for individual components of the state first (except abilities. It's hard to verify something as vague as abilities in the slightest).
- After all the state based pre-training is done, we freeze the weights and pre-train the attention and world model (to avoid cold start).
- After the whole pre-training process is done, we unfreeze the weights and let the model learn complex ideas associated with things in the embeddings during the training. For eg. learning that thunder wave causes paralysis or that U-turn attacks and causes a switch-out.

## Species
- Go over every possible pokemon and memorize the stats in the embeddings.
- Prediction of Type, HP, attack, defense, sp. attack, sp. defense, speed.
- Randomly arrange every pokemon and repeat in different epochs until the loss is very less.

## Types (Hierarchical Pre-training)
- The first (true) embedding. For pre-training, input is one type token, it will output 18 numbers (effectivenesses like 0x, 0.5x, 2x) against all 18 types.
- The second (MLP) embedding. For pre-training, input is the layer just before output from 3 instances of the pre-trained true embedding [move type, opponent type-1, opponent type-2]. The output is the specific effectiveness (for eg. fire against grass/steel will give 4x)
- Go over all 18 types (randomly arranged) and repeat in different epochs to memorize a gist of the types in the (true) embedding.
- Go over all possible combinations of [move type, opponent type-1, opponent type-2] (randomly arranged) and repeat in different epochs to memorize the in-practice type effectivenesses in the (MLP) embedding. move type and opponent type-1 must be proper types (not null/unk). opponent type-2 can be null (for single types)

## Moves
- Go over every possible move and memorize the stats in the embeddings.
- Prediction of type, power, PP , accuracy, is_special, is_physical, is_status, no. of hits (min and max).
- Randomly arrange every move and repeat in different epochs until the loss is very less.
