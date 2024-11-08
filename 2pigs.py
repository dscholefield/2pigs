import random

DEBUG = 0
adjust_delta = 5


def two_dice() -> tuple[int,int]:
    return (random.randint(1,6), random.randint(1,6))


def should_pass() -> int:
    return random.randint(1,100)


def adjust_weights(weights: tuple[float, ...]) -> tuple[float, ...]:
    tup_list = []
    for w in weights:
        adjust_value = float(random.randint(-adjust_delta, adjust_delta))
        tup_list.append(round(w + adjust_value, 2))
    return tuple(tup_list)


def debug_print(message: str) -> None:
    if DEBUG:
        print(message)


# For the nueral network we require two inputs: the current position
# and the weights for the neurons. The model will return a prediction
# of whether another roll is the best strategy for the current position
# expressed as a value where <50 will be pass, >=50 will be roll again

# The model is run from the perspective of making a prediction for player A

# Position is expressed as (turn, player A score, playerB score, turn score)
# The model pre-processes this position to use the following values (which
# are expressed as first layer input nuerons labeled [N.x]):
# - [N.0] turn (both players having played counts as a turn)
# - [N.1] turn is first turn (0 or 1)
# - [N.2] player A score
# - [N.3] player B score
# - [N.4] turn score
# - [N.5] difference between player A score and player B score (may be negative)
#     expressed as plyaer A - player B
# - [N.6] distance from player A score to 100
# - [N.7] distance from player B score to 100

# Mid layer neurons will combine weights from certain input layer neurons in 
# order to link their impact. These will include
# - [N.8] N.0 and N.2 and N.3 turn count and both players' score
# - [N.9] N.0 and N.2 and N.4 and N.6 turn count and 
#     player A's score, turn score and distance to 100

# Next layer neurons will combine from mid layer
# - [N.10] N.9 and N.3 player A's score, turn score and distance to 100 and
#     player B's score

# Output neuron (single)
# - [N.11] All nuerons 0-10 summed


def model(pos: tuple[int, int, int, int],
          weights: tuple[float, ...]) -> float:

    # preprocess the pos information
    layer1_neurons = [0, 0, 0, 0, 0, 0, 0, 0] 
    layer1_neurons[0] = pos[0]
    layer1_neurons[1] = 0
    if layer1_neurons[0] == 0:
        layer1_neurons[1] = 1
    layer1_neurons[2] = pos[1]
    layer1_neurons[3] = pos[2]
    layer1_neurons[4] = pos[3]
    layer1_neurons[5] = layer1_neurons[2] - layer1_neurons[3]
    layer1_neurons[6] = 100 - layer1_neurons[2]
    layer1_neurons[7] = 100 - layer1_neurons[3]

    # multiple layer1 neurons with weights
    for neuron in range(0,7):
        layer1_neurons[neuron] = layer1_neurons[neuron] * weights[neuron]
        debug_print(f"Value for neuron {neuron} is {layer1_neurons[neuron]}")

    # it's inelegant but we can express the unprocessed output
    # weight as a single calculation per layer 

    neuron_8 = (layer1_neurons[0] + layer1_neurons[2] + layer1_neurons[3]) * weights[8]
    neuron_9 = (layer1_neurons[0] + layer1_neurons[2] + layer1_neurons[4]) * weights[9]
    neuron_10 = (neuron_9 + layer1_neurons[3]) * weights[10]

    unprocessed_n11 = 0
    for neuron in (0,7):
        unprocessed_n11 = unprocessed_n11 + layer1_neurons[neuron]
    unprocessed_n11 = unprocessed_n11 + neuron_8 + neuron_9 + neuron_10

    return unprocessed_n11


# for position in [(50, 99, 99, 99), (8, 45, 56, 12), (0,0,0,0)]: 
#    weights = (0.05, 100.0, 0.05, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 1.0)
#    prediction = model(position, weights)
#    debug_print(f"Prediciton for position {position} is {prediction}")
# exit

# make two models play each other. The weights are what we are searching for
# the starting weights can be random but in practice, we can cut training time
# with something we believe will result in 'reasonble' play
player0_weights = (0.05, 10.0, -50, -0.5, -90, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05)
player1_weights = (-0.05, -10.0, +50, 0.5, 90, +0.05, +0.05, +0.05, -0.05, -0.05, -0.05)
# player1_weights = (0.1, 10, -20, -1, -85, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)

# keep a record of whether training is going very awry for a given player and
# return them to original if they lose 10 training rounds in a row

stored_player0_weights = player0_weights
stored_player1_weights = player1_weights
original_player0_weights = player0_weights
original_player1_weights = player1_weights
player0_best_weights = player0_weights
player1_best_weights = player1_weights
player0_best = 0
player1_best = 1
player0_run = 0
player1_run = 1


training_rounds = 20000
tround = 0
game_count = 500
player0_adjusts = 0
player1_adjusts = 0
player0_loss_run = 0
player1_loss_run = 0

for training_round in range(1, training_rounds):
    game_score = {0: 0, 1: 0}
    tround = tround + 1
    print(f"Training round {tround}")
    for game in range(1, game_count):
        debug_print(f"Playing game {game}")
        debug_print(f"Player game {game}")
        player_score = [0,0]
        player = 0
        debug_print(f"Player is {player}")
        turn_count = 0
        turn = 0
        debug_print(f"player score 0 is {player_score[0]}, player score 1 is {player_score[1]}")
        while player_score[0] < 100 and player_score[1] < 100 and turn < 200:
            debug_print(f"player score 0 is {player_score[0]}, player score 1 is {player_score[1]}")
            turn_count = turn_count + 1
            if turn_count % 2 == 0:
                turn = turn + 1
            turn_score = 0
            pass_turn = 0
            if player == 0:
                debug_print("changing to player 1")
                player = 1
            else:
                debug_print("changing to player 0")
                player = 0
            while pass_turn == 0:
                this_throw = two_dice()
                debug_print(f"throw is {this_throw[0]} and {this_throw[1]}")
                if this_throw[0] == 1 and this_throw[1] == 1:
                    debug_print("oops - two pigs thrown")
                    turn_score = 0
                    player_score[player] = 0
                    break
                if this_throw[0] == 1 or this_throw[1] == 1:
                    debug_print("oops - one pig thrown")
                    turn_score = 0
                    break
                turn_score = this_throw[0] + this_throw[1]
                if player_score[player]+turn_score >= 100:
                    debug_print(f"player {player} has won this game")
                    game_score[player] = game_score[player] + 1
                    debug_print(f"Player {player} has won")
                    player_score[player] = player_score[player] + turn_score
                    pass_turn = 1
                    break
                # time to run the model to decide whether to throw or not
                if player == 0:
                    position = (turn, player_score[0], player_score[1], turn_score) 
                    to_pass = model(position, player0_weights)
                else:
                    position = (turn, player_score[1], player_score[0], turn_score)
                    to_pass = model(position, player1_weights)
                debug_print(f"to pass score is {to_pass}")
                if to_pass < 50:
                    debug_print(f"Player {player} passing as less than 50")
                    pass_turn = 1
                    player_score[player] = player_score[player] + turn_score
                    debug_print(f"player score 0 is {player_score[0]}, player score 1 is {player_score[1]}")
                else:
                    debug_print(f"Player {player} elects to play")
        if turn >= 100:
            print("Caught in reset loop - escaping this game")
    # report on the end of the games played between the two models (players)
    # and adjust weights on loser and replay
    print(f"Player 0 {player0_weights} has scored {game_score[0]}\nPlayer 1 {player1_weights} has scored {game_score[1]}")
    # record these weights for both players if the scores are the best so far
    if game_score[0] > player0_best:
        player0_best_weights = player0_weights
        player0_best = game_score[0]
    if game_score[1] > player1_best:
        player1_best_weights = player1_weights
        player1_best = game_score[1]
    # look for long runs of losses and reset to best if stuck in a poor run 
    if game_score[0] < game_score[1]:
        player0_loss_run = player0_loss_run + 1
        player1_loss_run = 0
        stored_player0_weights = player0_weights
        player0_weights = adjust_weights(player0_weights)
        print(f"Adjusted weights on player 0 to {player0_weights}")
        player0_adjusts = player0_adjusts + 1
        if player0_loss_run > 5:
            player0_weights = player0_best_weights
            player0_loss_run = 0
            print("\n ADJUST player0 back to best after loss run \n")
    else:
        player1_loss_run = player1_loss_run + 1
        player0_loss_run = 0
        stored_player1_weights = player1_weights
        player1_weights = adjust_weights(player1_weights)
        print(f"Adjusted weights on player 1 to {player1_weights}")
        player1_adjusts = player1_adjusts + 1
        if player1_loss_run > 5:
            player1_weights = player1_best_weights
            player1_loss_run = 0
            print("\n ADJUST player1 back to best after loss run\n")
print(f"\nPlayer 0 has been adjusted {player0_adjusts} times, and player 1 {player1_adjusts} times.")
print(f"Original model for player0 was {original_player0_weights}")
print(f"Original model for player1 was {original_player1_weights}")
print(f"Best model for player0 was {player0_best_weights}")
print(f"Best model for player1 was {player1_best_weights}")
print(f"{training_rounds} training rounds of {game_count} games each - {training_rounds * game_count} games played\n\n")