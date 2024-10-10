# popular dice game (known as Yathzee) played by a neural network

--- work in progress ---


## Synopsis

### Game Representation

One round of Yahtzee consists of 13 turns, because there are 13 possible categories on the scorecard. At the start of a turn, five dice are rolled. These are represented as an array of 5 integers, e.g. `[3, 4, 2, 6, 1]`. For the model, they are then one-hot encoded into a binary array. With the previous example, the contents of this array would look like `00100 000100 010000 000001 100000` (spaces added for readability). The model then can make the decision of keeping all dice, or re-roll any of the dice. This is repeated another time, so after the second re-roll the dice have to be kept as-is.

The 13 categories on the scorecard are also represented as a binary array, with a `1` for any category that has already been taken, and `0` for the rest. For example, if the model decides to take the previous roll for a small straight, the scorecard input for the next round would look like this `000000 0001000` because the value at index `9` represents a small straight.

Here are all the categories on the scorecard with their indices:

| Index | Name              | Description                                     | Score               |
|-------|-------------------|-------------------------------------------------|---------------------|
| 0     | Aces              | Any number of Ones                              | The sum of dice with the number 1|
| 1     | Twos              | Any number of Twos                              | The sum of dice with the number 2|
| 2     | Threes            | Any number of Threes                            | The sum of dice with the number 3|
| 3     | Fours             | Any number of Fours                             | The sum of dice with the number 4|
| 4     | Fives             | Any number of Fives                             | The sum of dice with the number 5|
| 5     | Sixes             | Any number of Sixes                             | The sum of dice with the number 6|
| 6     | Three of a kind   | At least three dice the same, Sum of all Dice   | Sum of all Dice      |
| 7     | Four of a kind    | At least four dice the same, Sum of all Dice    | Sum of all Dice      |
| 8     | Full House        | Three of one number and two of another          | 25                  |
| 9     | Small Straight    | Four sequential dice                            | 30                  |
| 10    | Large Straight    | Five sequential dice                            | 40                  |
| 11    | Yahtzee           | All Five Dice the Same                          | 50                  |
| 12    | Chance            | Any, Sum of all dice                            | Sum of all Dice      |

<br>
The last value of the model input is the number of re-rolls left, which is either 2, 1 or 1. In total, the model gets an one-dimensional input vector of size 44.


### Model Architecture



### Hyperparameters



### Training Loop

