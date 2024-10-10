import sys
import os
from shutil import copy2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Bernoulli, Categorical
from torchinfo import summary
import yaml
import numpy as np
from tqdm import tqdm
import logging

from torch_utils import prepare_input, reroll_dice, initialize_weights
from rules import validate_dice_categories
from score import calculate_score, calculate_total_score
from utils import plot_policy_losses, plot_game_scores, format_input, format_score_action


logging.basicConfig(
        level=logging.DEBUG,
        # level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s", datefmt="%d-%b-%y %H:%M:%S",
        handlers=[
            logging.FileHandler("debug.log", mode="w"),
            logging.StreamHandler(sys.stdout)
        ]
    )

logging.debug(__file__)


class DiceNetwork(nn.Module):
    def __init__(self):
        super(DiceNetwork, self).__init__()

        device = torch.cuda.get_device_name(torch.cuda.current_device())
        print(f"The model is running on {device}")

        # Shared layers
        self.fc1 = nn.Linear(44, 128)  # Input: dice (30), score info (13), rolls left (1)
        self.fc2 = nn.Linear(128, 128)  # Shared hidden layer
        self.fc3 = nn.Linear(128, 128)  # Another shared hidden layer
        
        # Dice re-roll head (outputs 5 binary decisions for dice)
        self.dice_output = nn.Linear(128, 5)  # 5 outputs for re-rolling dice
        
        # Score box head (outputs 13 categorical probabilities for scoring)
        self.score_output = nn.Linear(128, 13)  # 13 outputs for score box decision


    def forward(self, x, rolls_left):
        # Shared layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        
        # Dice re-roll decision: Sigmoid activation for binary output (only if rolls left > 0)
        if rolls_left > 0:
            dice_decisions = torch.sigmoid(self.dice_output(x))  # Re-roll decision
        else:
            dice_decisions = None  # No re-roll decision when rolls left is 0
      
        # Score category decision: Softmax activation for categorical output
        score_decision = torch.softmax(self.score_output(x), dim=0)  # Score category decision
        
        return dice_decisions, score_decision


def compute_discounted_rewards(rewards: list[int], gamma: float) -> list[float]:
    """
    Compute the discounted cumulative rewards for the episode.
    """
    discounted_rewards = []
    cumulative_reward = 0
    for reward in reversed(rewards):
        cumulative_reward = reward + gamma * cumulative_reward
        discounted_rewards.insert(0, cumulative_reward)
    return discounted_rewards


def train_model(model: torch.nn.Module, optimizer: torch.optim.Optimizer, settings):
    # training loop
    losses = []
    scores = []

    for episode in tqdm(
        range(settings["training_episodes"]), 
        disable=logging.getLogger().isEnabledFor(logging.DEBUG)
        ):
        logging.debug("############################ Starting Episode %s ############################" % (episode + 1))

        rewards = []  # store the rewards for actions based on the score sheet
        log_probs = []  # store the log probabilities of actions for reward calculation
        entropy_terms = []  # store mean entropy of actions to calculate the total entropy

        # Initialize game state
        score_categories = [0] * 13  # Score card, all categories unfilled (either 0 or 1)
        scores_achieved = [0] * 13  # Store the actual scores after each turn
        
        for _ in range(13):  # 13 turns in Yahtzee
            score = 0
            rolls_left = 2
            dice = torch.randint(1, 7, (5,))
            logging.debug("initial roll: %s" % dice)

            log_probs_turn = []  # Store log probabilities for the current turn

            # While there are rolls left, make re-roll decisions
            while rolls_left > 0:
                logging.debug("rerolls left: %s" % rolls_left)

                input_vector = prepare_input(dice, score_categories, rolls_left)
                dice_decisions, score_decisions = model(input_vector, rolls_left)

                logging.debug("input vector: %s" % format_input(input_vector))
                logging.debug("dice decisions: %s" % dice_decisions)
                logging.debug("score decisions: %s" % score_decisions)
                
                # Sample action (re-roll decision) from the policy's probability distribution
                dice_action_dist = Bernoulli(dice_decisions)
                # Collect entropy for dice decisions (higher means more uncertainty)
                entropy_terms.append(dice_action_dist.entropy().mean())
                # Sample action based on policy
                dice_action = dice_action_dist.sample()

                logging.debug("dice action: %s" % dice_action)
                
                log_probs_turn.append(dice_action_dist.log_prob(dice_action).sum())

                # Apply dice re-roll decision
                if dice_action.sum() == 0:
                    # Model decided to keep all dice, break early
                    logging.debug("Model keeps the dice.")
                    break
                dice = reroll_dice(dice, dice_action.detach().numpy())  # Re-roll dice
                logging.debug("reroll result: %s" % dice)
                rolls_left -= 1
            
            # After the final roll, select a score category
            _, score_decisions = model(input_vector, rolls_left=0)
            score_action_dist = Categorical(score_decisions)
            # Collect entropy for score category selection
            entropy_terms.append(score_action_dist.entropy().mean())
            # Sample score category
            score_action = score_action_dist.sample() 
            
            # Store log prob for backprop
            log_probs_turn.append(score_action_dist.log_prob(score_action))  
            
            # Get valid categories and ensure action is valid
            valid_categories = validate_dice_categories(dice.numpy(), score_categories)
            logging.debug("valid categories: %s" % "".join([str(x) for x in valid_categories]))
            score_decision_idx = score_action.item()
            
            if valid_categories[score_decision_idx] == 1:
                # Apply score category and calculate score
                score = calculate_score(dice, score_decision_idx) 
                score_categories[score_decision_idx] = 1  # Mark category as filled
            else:
                # If there are no valid categories left, we need to "cross out" a category
                if np.sum(valid_categories) == 0:
                    logging.debug("No valid categories left. Choosing a random unfilled category to cross out.")
                    # TODO: can the model be trained to choose a category here?
                    unfilled_categories = [i for i, filled in enumerate(score_categories) if filled == 0]
                    # Randomly pick a remaining category
                    score_decision_idx = np.random.choice(unfilled_categories)
                    score = -10  # negative Score as penalty
                else:
                    # Select the valid category with the highest predicted score
                    valid_scores = np.where(np.array(valid_categories) == 1, score_decisions.detach().numpy(), -np.inf)
                    score_decision_idx = np.argmax(valid_scores)
                
                    # Apply the score decision and calculate the score
                    score = calculate_score(dice, score_decision_idx)

                # Mark the category as filled
                score_categories[score_decision_idx] = 1

            if score > 0:
                logging.debug("Model decides to choose %s with dice %s. Resulting score: %s" % (
                    format_score_action(score_decision_idx), dice.numpy(), score
                ))
            else:
                logging.debug("Model decides to cross out %s" % (format_score_action(score_decision_idx)))
            # logging.debug("score decision: %s (%s)" % (score_decision_idx, format_score_action(score_decision_idx)))
            # logging.debug("Score: %s" % score)
            logging.debug("scores taken after this turn: %s" % "".join([str(x) for x in score_categories]))
        
            # Replicate the reward for each action taken in this turn (re-rolls + score decision)
            # reward is the score of the chosen category, or -10 if a category was crossed out
            for _ in log_probs_turn:
                rewards.append(score)
            
            # Add the log probabilities of all actions from this turn to the main list
            log_probs.extend(log_probs_turn)

            # store the score for later calculations
            scores_achieved[score_decision_idx] = max(score, 0)

        # compute the total score for this sheet
        total_score = calculate_total_score(scores_achieved)

        # add the total score to all rewards
        rewards = [r + total_score for r in rewards]
        
        # Compute discounted rewards
        discounted_rewards = compute_discounted_rewards(rewards, settings["gamma"])
        logging.debug("discounted_rewards: %s" % discounted_rewards)
        
        # Normalize rewards for stability
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        
        # Compute the policy gradient loss
        policy_loss = []
        logging.debug("computing policy loss")
        for log_prob, reward in zip(log_probs, discounted_rewards):
            # Multiply log-probability by the discounted reward
            policy_loss.append(-log_prob * reward)

        policy_loss = torch.stack(policy_loss).sum()

        # Calculate the total entropy (summed over all actions) to encourage exploration
        entropy_term = torch.stack(entropy_terms).mean()
        policy_loss = policy_loss - float(settings["entropy_coefficient"]) * entropy_term

        logging.debug("policy loss: %s" % policy_loss.item())
        
        # Backpropagate
        optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # append losses and scores for plotting later
        losses.append(policy_loss.detach().item())
        scores.append(total_score)

        logging.debug("Sum of rewards: %s" % sum(rewards))
        logging.debug("Episode %s: Total score: %s" % (episode + 1, total_score))

    # store the results

    # TODO: create a output/training folder
    if not os.path.exists("output/training"):
        os.makedirs("output/training")
    
    # check for previous runs and get the index for the next run
    try:
        next_run_idx = sorted([int(d.split("run")[1]) for d in os.listdir("output/training")])[-1] + 1
    except IndexError:
        next_run_idx = 0
    # create a run0, run1, ... folder
    run_folder = f"output/training/run{next_run_idx}"
    os.makedirs(run_folder)

    plot_policy_losses(losses, run_folder)
    plot_game_scores(scores, run_folder)

    # save model settings
    copy2('settings.yaml', os.path.join(run_folder, "settings.yaml"))


def get_optimizer(model: torch.nn.Module, settings: dict) -> optim.Optimizer:
    return optim.Adam(model.parameters(), lr=float(settings["learning_rate"]))


if __name__ == "__main__":

    with open('settings.yaml', 'r') as file:
        settings = yaml.safe_load(file)

    model = DiceNetwork()
    summary(model, input_data={"x": prepare_input([1] * 5, [0] * 13, 2), "rolls_left": 2})

    optimizer = get_optimizer(model, settings)

    model.apply(initialize_weights)  # Xavier initialization (probably unnecessary)

    train_model(model, optimizer, settings)