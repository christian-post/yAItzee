import torch
import numpy as np


def prepare_input(dice: list[int], score_categories: list[int], rolls_left: int):
    """
    Convert the current game state into a flattened input vector for the model.
    Combines dice values, score category status, and rolls left.
    """
    # One-hot encode the dice
    dice_one_hot = torch.zeros(5, 6)  # 5 dice, 6 possible values per die
    for i, value in enumerate(dice):
        dice_one_hot[i, value - 1] = 1  # -1 because dice values are from 1 to 6
    
    # Flatten the dice one-hot encoding and combine with score categories and rolls left
    return torch.cat([dice_one_hot.flatten(),
                      torch.tensor(score_categories, dtype=torch.float),
                      torch.tensor([rolls_left], dtype=torch.float)])


def reroll_dice(dice: torch.Tensor, decisions: np.array | list[int]):
    """
    Re-roll the dice based on the re-roll decisions.
    
    Parameters:
    dice (tensor): The current dice values.
    decisions (list or array): A binary list indicating which dice to re-roll (1 to re-roll, 0 to hold).
    
    Returns:
    new_dice (tensor): The dice after re-rolling.
    """
    for i in range(len(dice)):
        if decisions[i] == 1:  # If decision is 1, re-roll this die
            dice[i] = torch.randint(1, 7, (1,)).item()
    return dice