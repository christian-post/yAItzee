import numpy as np
from collections import Counter
import torch


def is_small_straight(dice: list[int]) -> bool:
    """Check if the dice form a small straight (sequence of 4 consecutive numbers)."""
    unique_dice = sorted(set(dice))
    straights = [{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}]
    return any(straight.issubset(unique_dice) for straight in straights)


def is_large_straight(dice: list[int]) -> bool:
    """Check if the dice form a large straight (sequence of 5 consecutive numbers)."""
    unique_dice = sorted(set(dice))
    return unique_dice == [1, 2, 3, 4, 5] or unique_dice == [2, 3, 4, 5, 6]


def validate_dice_categories(dice: list[int], score_categories: list[int]) -> list[int]:
    """
    Validate the current dice roll against all 13 score categories.
    
    Parameters:
    dice (list): A list of 5 integers representing the dice roll (values between 1 and 6).
    score_categories (list): A binary list of length 13 indicating which score categories are already filled (1 means filled, 0 means unfilled).
    
    Returns:
    valid_categories (list): A binary list of length 13 where 1 means the category is valid for the current dice roll, and 0 means it is not.
    """
    # Initialize a list to store validity (1 = valid, 0 = not valid)
    valid_categories = [0] * 13

    # Count the occurrences of each die value
    dice_counter = Counter(dice)
    
    # Upper section validation (Ones to Sixes)
    for i in range(6):
        if dice_counter[i + 1] > 0 and score_categories[i] == 0:
            valid_categories[i] = 1  # Valid if the dice show at least one of that number
    
    # Three of a Kind (at least 3 dice of the same value)
    if max(dice_counter.values()) >= 3 and score_categories[6] == 0:
        valid_categories[6] = 1
    
    # Four of a Kind (at least 4 dice of the same value)
    if max(dice_counter.values()) >= 4 and score_categories[7] == 0:
        valid_categories[7] = 1
    
    # Full House (three of one number and two of another)
    if sorted(dice_counter.values()) == [2, 3] and score_categories[8] == 0:
        valid_categories[8] = 1
    
    # Small Straight (sequence of 4 consecutive numbers)
    if is_small_straight(dice) and score_categories[9] == 0:
        valid_categories[9] = 1
    
    # Large Straight (sequence of 5 consecutive numbers)
    if is_large_straight(dice) and score_categories[10] == 0:
        valid_categories[10] = 1
    
    # Yahtzee (all 5 dice show the same number)
    if max(dice_counter.values()) == 5 and score_categories[11] == 0:
        valid_categories[11] = 1
    
    # Chance (always valid)
    if score_categories[12] == 0:
        valid_categories[12] = 1
    
    return valid_categories


def validate_dice_reroll(decision: torch.tensor | np.array | list[int], rolls_left: int) -> bool:
    """
    Validate if the dice re-roll decision is valid based on the number of rolls left.
    
    Parameters:
    decision (tensor): A tensor of 5 binary values indicating whether to re-roll each die.
    rolls_left (int): Number of rolls left (1 or 2).
    
    Returns:
    valid (bool): True if the decision is valid, False otherwise.
    """
    # Re-roll is valid only if there are rolls left
    if rolls_left == 0:
        return False
    
    # Ensure decision has correct dimensions (5 dice)
    if len(decision) != 5:
        return False
    
    # Since re-rolling any subset of dice is valid, we return True if the above conditions are met
    return True


def validate_score_category(selection: int, available_categories: list[int]) -> bool:
    """
    Validate if the score category selected is available.
    
    Parameters:
    selection (int): The index of the selected score category (0-12).
    available_categories (list or array): A binary list of length 13 where 0 indicates the category is available, and 1 indicates it is filled.
    
    Returns:
    valid (bool): True if the score category is valid (i.e., unfilled), False otherwise.
    """
    # Ensure selection is within valid bounds (0 to 12 for 13 categories)
    if selection < 0 or selection >= len(available_categories):
        return False
    
    # Check if the selected category is still available (0 means available)
    return available_categories[selection] == 0