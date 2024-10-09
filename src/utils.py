import matplotlib
import matplotlib.pyplot as plt
import logging
import torch
import numpy as np
from typing import Union


logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

def plot_policy_losses(losses: list[float]):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Policy Loss', color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Policy Loss')
    plt.title('Policy Loss over Training Episodes')
    plt.grid(True)
    plt.legend()
    plt.savefig("output/training_loss.png")


def plot_game_scores(scores: list[float]):
    plt.figure(figsize=(10, 6))
    plt.plot(scores, label='Game Score', color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Game Score')
    plt.title('Game Score over Training Episodes')
    plt.grid(True)
    plt.legend()
    plt.savefig("output/training_scores.png")


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