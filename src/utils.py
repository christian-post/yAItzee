import os
import matplotlib
import matplotlib.pyplot as plt
import logging
import torch
import numpy as np
from typing import Union


logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)


def plot_results(y: list[float], **kwargs) -> None:
    """base function for the other plotting functions"""
    plt.figure(figsize=(10, 6))

    if len(y) > 100:
        # Bin the losses into 100 bins
        bins = np.array_split(y, 100)
        bin_means = [np.mean(bin_) for bin_ in bins]

        # Compute the x-axis values (the average episode number for each bin)
        bin_indices = np.array_split(range(len(y)), 100)
        x_vals = [np.mean(indices) for indices in bin_indices]

        # Plot vertical lines for all values in each bin (semi-transparent)
        for i, bin_ in enumerate(bins):
            plt.vlines(x_vals[i], min(bin_), max(bin_), kwargs["color"], alpha=0.4)

        # Plot the mean of each bin (opaque)
        plt.plot(x_vals, bin_means, label=kwargs["label"], color=kwargs["color"], alpha=1.0)
    else:
        # If less than or equal to 100, plot all values as usual
        plt.plot(y, label=kwargs["label"], color=kwargs["color"])

    plt.xlabel(kwargs["xlabel"])
    plt.ylabel(kwargs["ylabel"])
    plt.title(kwargs["title"])
    plt.grid(True)
    plt.legend()
    plt.savefig(kwargs["filename"])


def plot_policy_losses(losses: list[float], folder: str):
    plot_results(
        y=losses,
        label="Policy Loss",
        color="blue", 
        xlabel="Episode",
        ylabel="Policy Loss", 
        title="Policy Loss over Training Episodes",
        filename=os.path.join(folder, "training_loss.png")
    )


def plot_game_scores(scores: list[float], folder: str):
    plot_results(
        y=scores, 
        label="Game Score", 
        color="red", 
        xlabel="Episode", 
        ylabel="Game Score", 
        title="Game Score over Training Episodes",
        filename=os.path.join(folder, "training_scores.png")
    )


def format_input(input_vector: Union[torch.Tensor, np.array, list[int]]) -> str:
    if type(input_vector) == torch.tensor:
        input_vector = input_vector.tolist()

    raw_str = "".join(map(lambda x: str(int(x)), input_vector))

    # insert spaces for readability
    # first 30 are one-hot encoded dice
    # then 13 for the scores
    # last one is the number of rolls left
    space_indices = [6, 12, 18, 24, 30, 43]
    offset = 0
    for index in space_indices:
        adjusted_index = index + offset
        raw_str = raw_str[:adjusted_index] + " " + raw_str[adjusted_index:]
        offset += 1  # Increment the offset since a space is inserted

    return raw_str


def format_score_action(index: int) -> str:
    """
    Translates the score index to a human readable info
    
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

    info_strings = [
        "Aces", "Twos", "Threes", "Fours", "Fives", "Sixes",
        "Three of a kind", "Four of a kind", "Full House",
        "Small Straight", "Large Straight", "Yahtzee", "Chance"
    ]

    return info_strings[index]