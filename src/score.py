from collections import Counter
import torch
import numpy as np
from typing import Union



def calculate_score(dice: Union[torch.Tensor, np.array, list[int]], category: int) -> int:
    """
    Calculate the score for the selected category based on the current dice.
    
    Parameters:
    dice (list or array): A list of 5 integers representing the current dice roll (values between 1 and 6).
    category (int): The index of the category selected (0-12).
    
    Returns:
    int: The calculated score for the selected category.
    """
    if type(dice) == torch.Tensor:
        dice = dice.detach().numpy()
    
    # Count occurrences of each die value
    dice_counter = Counter(dice)
    
    # Upper Section: Ones (0) to Sixes (5)
    if category in range(6):
        dice_value = category + 1  # Ones is category 0, so we add 1 to get the value (1-6)
        return dice_counter[dice_value] * dice_value  # Sum of dice showing this value
    
    # Three of a Kind (6): At least three dice showing the same value
    elif category == 6:
        if max(dice_counter.values()) >= 3:
            return sum(dice)  # Sum of all dice
        else:
            return 0
    
    # Four of a Kind (7): At least four dice showing the same value
    elif category == 7:
        if max(dice_counter.values()) >= 4:
            return sum(dice)  # Sum of all dice
        else:
            return 0
    
    # Full House (8): Three of one number and two of another
    elif category == 8:
        if sorted(dice_counter.values()) == [2, 3]:
            return 25
        else:
            return 0
    
    # Small Straight (9): Sequence of 4 consecutive numbers (e.g., 1-2-3-4)
    elif category == 9:
        unique_dice = sorted(set(dice))
        straights = [{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}]
        if any(straight.issubset(unique_dice) for straight in straights):
            return 30
        else:
            return 0
    
    # Large Straight (10): Sequence of 5 consecutive numbers (e.g., 1-2-3-4-5)
    elif category == 10:
        unique_dice = sorted(set(dice))
        if unique_dice == [1, 2, 3, 4, 5] or unique_dice == [2, 3, 4, 5, 6]:
            return 40
        else:
            return 0
    
    # Yahtzee (11): All five dice showing the same number
    elif category == 11:
        if max(dice_counter.values()) == 5:
            return 50
        else:
            return 0
    
    # Chance (12): Sum of all dice, no restrictions
    elif category == 12:
        return sum(dice)
    
    # Invalid category, return 0
    return 0



def calculate_total_score(scores_achieved: list[int]) -> int:
    """
    Sums up all achieved scores on the scorecard and checks for the bonus points

    Parameters:
    scores_achieved: list[int]: A complete list of scores achieved in one round.

    Returns:
    Total Score
    """
    # TODO: return the bonus as well for better rewards?

    bonus = 0
    total_score = sum(scores_achieved)

    if sum(scores_achieved[:5]) >= 63:
        bonus = 35
        total_score += bonus

    return total_score