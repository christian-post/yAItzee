import yaml
from model import DiceNetwork, get_optimizer, initialize_weights, train_model, print_summary


if __name__ == "__main__":
    with open('settings.yaml', 'r') as file:
        settings = yaml.safe_load(file)

    model = DiceNetwork(settings)
    optimizer = get_optimizer(model, settings)

    print_summary(model)

    model.apply(initialize_weights)  # Xavier initialization (probably unnecessary)

    train_model(model, optimizer, settings)

