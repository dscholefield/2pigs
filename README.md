# 2pigs
A simple neural net for playing the dice game '2 pigs'.

## Introduction
This AI model has been developed as a demonstration of how to create a simple net with up to 4 layers, and how to train that net to play a dice game called '2 pigs'.

Because it's a demonstration model, it doesn't use an off the shelf pre-trained network, or libraries, or sophisticated training algorithms based on gradient descent etc. Instead it's a small, hand built, set of neurons that are trained through a simple mechanism of self-reinforcement combined with controlled random mutation in the weights. Nobody would use this approach in order to find the optimal, or even near optimal, model weights, but it is good enough to both play the game (quite well!) and explain the principles of machine learning. 

## Rules of the Game
In 2 pigs, two players take it in turns to roll 2 dice. If they don't roll a '1' on either dice then they can either bank the total showing on the two dice and end their turn, or they can elect to throw the dice again. Every dice throw in that turn is added together until either the player decides to end their turn, or they throw a '1' on either dice. If they throw a single '1' then they lose their score for that turn and have to pass the dice to the other player. If they role a double '1' then they not only lose any score they have made in the current turn, but their overall game score. 

The winner is the first player to reach a score of 100.

## About the Model
The model for playing this game is defined in the function 'model()' in the file '2pigs.py'. It takes two tuples as input: the first represents the position, and the second represents the weights for 10 of the 11 neurons in the net. There is no limit on the value of the weights but in training it appears that they may change from between -100 and +100 due to the adjustment algorithm during the training rounds (the variable 'adjust_delta' determines the range of values each weight may be changed by between training rounds and therefore determines the search space. There has been no experimentation with this value so far to determine whether a larger or small delta effects the speed of the search for optimal weights.)

The model returns a float value which is the prediction of the model as to whether the player, in that position, should elect to roll the dice again.

Arbitrarily, in the training code, the score of 50.0 is chosen as the decision point to throw or not to throw, so a score from the model() function of 50 or greater means that the player is advised to throw again, and less than 50 the player should pass. 

This value is arbitrary but whichever value is chosen should be used in both the training phase and any playing phase or the prediction is not valid.

The position value passed into the model() function has 4 separate parts:
    - the current turn value (from 1 to any integer)
    - the current score for player 0
    - the current score for player 1
    - the current turn score for player 0

Note that the model() function does not have any state so if you are playing a game between two players, both of which are using the model, you need to swap the scores for the two players before calling the model() function.

The weights value passed into the model() function is a list of 10 floats.

Thus, the function call model((3, 65, 72, 18), (0.5, 7, 19, -5.5, 40, 0.5, 0.5, 0.5, -2, -8)) will return a prediction on whether the player 0 should roll or not given that the turn number is '3', the player that the prediction is being made about has a current score of 65, the opponent has a current score of 72, and the player that the prediction is being made about has a current turn score of 18. 

If the function returns a value of, for example, 117.67 then this is an indicator that they should roll again and risk the rolling of a '1' or double '1', but an example score of 38.991 suggests that they should bank their score of 18 and pass the dice to the other player. 

The file '2pigs.py' hard codes the initial weights for two players. The value of these weights isn't important as they are a starting point for training and will be adjusted based on the outcome of games where the model 'plays itself' many times.

## Training the Model
The file '2pigs.py' will then attempt to find a good (almost certainly not optimal) set of weights for use in real play by playing against itself with two players: the first with one of the initial hard coded set of weights, and the other with the other set of hard coded weights.

The variable 'training_rounds' determines how many times the training goes around the loop of setting each player against each other, and the variable 'game_count' determines how many games two players with the same weights play before the training algorithm determines which set of weights is 'the best' i.e. has won the majority of games in a round.

For each round, say with 500 games played in a round, the losing player's set of weights are adjusted by randomly increasing or decreasing each weight by a small amount (the size of the change being determined by the 'adjust_delta' discussed above.) The two players now go into the next training round with the new weights for the losing player having been adjusted. The rounds continue in this fashion with the losing player's weights being adjusted for each round. This means that after all of the training rounds, the most effective set of weights will 'win'.

Unfortunately the training is a little more complex for two reasons:

- without any additional controls, a very badly losing set of weights enters a spiral of descent whereby no matter how much adjustment has been made for each round, it continues to lose badly and has no way out of the ever increasing error delta
- the game itself may never end (or run for ever) due to the number of '1's that are thrown, which may cause scores to be regularly reset ad-infinitum. 

The training algorithm deals with these two outcomes by:

- if a set of weights loses a long sequence of games unbroken by any wins, its weights are returned to the last 'best weights' so that it can adjust into a new set that is hopefully outside of the n-dimensional gradient of doom!
- if a game continues on for a very large number of turns (indicating that the weights for both players are encouraging risky behaviour that results in constant '1's being thrown in long sequences of throws in a single turn) then it is terminated early and the weights adjusted accordingly

It's worth noting that this training algorithm is very simplistic and although the default training session is set to be 20000 rounds of 500 games per round (10 million games played in all) this is a very small fraction of the search space for all possible weight combinations, and there is no way to know whether the final winning weights are even close to optimal. 

## Playing Against the Model
There is a second file in the repo: 'play_2pigs.py' which enables you to play against the trained AI. The weights are hard coded into the variable 'player0_weights' which you can copy from the output of the training. Then, starting this script will enable a human to play against the model. Up to this point, the AI model has consistently beaten me, so something is working in the training (or I'm just a very poor games player!) 

## Training Performance
Currently, on an M3 Macbook Pro, using a single thread, the training alogorithm will play 10 million games in about 30 minutes. This could be greatly speeded up by multithreading (although it wouldn't make much more of a dent in the search space!)