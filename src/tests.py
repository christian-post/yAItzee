from torch_utils import prepare_input, reroll_dice, initialize_weights
from rules import validate_dice_categories
from score import calculate_score




if __name__ == "__main__":
    # --- dice roll validation ---

    """
    0 Aces = Any, The sum of dice with the number 1
    1 Twos = Any, The sum of dice with the number 2
    2 Threes = Any, The sum of dice with the number 3
    3 Fours = Any, The sum of dice with the number 4
    4 Fives = Any, The sum of dice with the number 5 
    5 Sixes = Any, The sum of dice with the number 6

    6 Three of a kind = At least three dice the same, Sum of all Dice
    7 four of a kind = At least four dice the same, Sum of all Dice
    8 Full House = Three of one number and two of another, 25
    9 Small Straight = Four sequential dice, 30
    10 Large Straight = Five sequential dice, 40
    11 Yahtzee, All Five Dice the Same, 50
    12 Chance = Any, Sum of all dice
    """

    score_categories = [0] * 13

    # Large Straight
    dice = [1, 2, 3, 4, 5]
    result = validate_dice_categories(dice, score_categories)
    assert result == [1, 1, 1, 1, 1, 0,
                      0, 0, 0, 1, 1, 0, 1]
    
    # Small Straight
    dice = [2, 3, 4, 5, 2]
    result = validate_dice_categories(dice, score_categories)
    assert result == [0, 1, 1, 1, 1, 0,
                      0, 0, 0, 1, 0, 0, 1]
    
    # Yahtzee
    dice = [1, 1, 1, 1, 1]
    result = validate_dice_categories(dice, score_categories)
    assert result == [1, 0, 0, 0, 0, 0,
                      1, 1, 0, 0, 0, 1, 1]
    
    
