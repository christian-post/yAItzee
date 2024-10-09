import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Bernoulli, Categorical
import yaml
import numpy as np

from torch_utils import prepare_input, reroll_dice
from rules import validate_dice_categories
from score import calculate_score


class DiceNetwork(nn.Module):
    def __init__(self):
        super(DiceNetwork, self).__init__()
        
        # Shared layers
        self.fc1 = nn.Linear(44, 128)  # Input: dice (30), score info (26), rolls left (1)
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


def compute_discounted_rewards(rewards: list[float], gamma: float) -> list[float]:
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
    for episode in range(settings["training_episodes"]):
        print(f"Starting Episode {episode + 1}")
        rewards = []
        log_probs = []

        # Initialize game state
        rolls_left = 2
        dice = torch.randint(1, 7, (5,))
        score_categories = [0] * 13  # All categories unfilled
        
        total_score = 0
        
        for _ in range(13):  # 13 turns in Yahtzee
            # While there are rolls left, make re-roll decisions
            while rolls_left > 0:
                input_vector = prepare_input(dice, score_categories, rolls_left)
                dice_decisions, score_decision = model(input_vector, rolls_left)
                
                # Sample action (re-roll decision) from the policy's probability distribution
                dice_action_dist = Bernoulli(dice_decisions)
                dice_action = dice_action_dist.sample()  # Sample action based on policy
                
                log_probs.append(dice_action_dist.log_prob(dice_action))  # Store log prob for backprop

                print(dice_action.detach().numpy())
                
                # Apply dice re-roll decision
                if dice_action.sum() == 0:
                    # Model decided to keep all dice, break early
                    break
                dice = reroll_dice(dice, dice_action.detach().numpy())  # Re-roll dice
                rolls_left -= 1
            
            # After the final roll, select a score category
            _, score_decision = model(input_vector, rolls_left=0)
            score_action_dist = Categorical(score_decision)
            score_action = score_action_dist.sample()  # Sample score category
            
            log_probs.append(score_action_dist.log_prob(score_action))  # Store log prob for backprop
            
            # Get valid categories and ensure action is valid
            valid_categories = validate_dice_categories(dice.numpy(), score_categories)
            score_decision_idx = score_action.item()
            
            if valid_categories[score_decision_idx] == 1:
                # Apply score category and calculate score
                total_score += calculate_score(dice, score_decision_idx) 
                score_categories[score_decision_idx] = 1  # Mark category as filled
            else:
                # Select a valid category with highest predicted value
                valid_scores = np.where(np.array(valid_categories) == 1, score_decision.detach().numpy(), -np.inf)
                score_decision_idx = np.argmax(valid_scores)
                total_score += calculate_score(dice, score_decision_idx)
                score_categories[score_decision_idx] = 1  # Mark category as filled
        
        # Store the final reward (total score for the episode)
        rewards.append(total_score)
        
        # Compute discounted rewards
        discounted_rewards = compute_discounted_rewards(rewards, settings["gamma"])
        
        # Normalize rewards for stability
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        
        # Compute the policy gradient loss
        policy_loss = []
        for log_prob, reward in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)  # Multiply log-probability by the discounted reward
        policy_loss = torch.cat(policy_loss).sum()
        
        # Backpropagate
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        print(f"Episode {episode + 1}: Total score: {total_score}")



if __name__ == "__main__":

    with open('settings.yaml', 'r') as file:
        settings = yaml.safe_load(file)

    model = DiceNetwork()
    optimizer = optim.Adam(model.parameters(), lr=settings["learning_rate"])

    train_model(model, optimizer, settings)