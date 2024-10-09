import matplotlib.pyplot as plt



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