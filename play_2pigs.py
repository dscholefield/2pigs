import random
import time
import readchar
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style


DEBUG = 0
adjust_delta = 5


def banner():
    print(f"{Fore.RED}" + 
        """                             
                                  
 @@@@@                     #@@@@@ 
 @      @  @@        @@ @.      @ 
 @ @@@@@@               +@@@@@  @ 
  @ @@@      2 PIGS!!      @@@ @  
   @ @                      @ @   
    @       @@     @@@       @    
   @        @@@    @@@            
   @                          @   
   @         @@::::@@         @   
   @       @::::::::::@       @   
   %      @::@@@::@@*::@     -    
    @     @:::@-::@@:::           
     @      @::::::::@     -      
       @              @   @       
         @-     @@@@@  @#         
             *@@@@@@              

 ++ D Scholefield. Ver 1.0 ++
          """
    )
    print(Style.RESET_ALL)


def two_dice() -> tuple[int,int]:
    return (random.randint(1,6), random.randint(1,6))


def should_pass() -> int:
    return random.randint(1,100)


def debug_print(message: str) -> None:
    if DEBUG:
        print(message)


def human(pos: tuple[int, int, int, int]) -> int:
    input_char = ''
    return_weight = 0
    print(f"Your game score: {pos[1]}, their game score {pos[2]}, turn score so far {pos[3]}")
    print("Press 'p' to pass and 'r' to roll...")
    while input_char != 'p' and input_char != 'r':
        input_char = readchar.readchar()
    if input_char == 'p':
        return_weight = 49
    else:
        return_weight = 51
    return return_weight  


def model(position: tuple[int, int, int, int],
          weights: tuple[float, ...]):

    # preprocess the position information
    layer1_neurons = [0, 0, 0, 0, 0, 0, 0, 0] 
    layer1_neurons[0] = position[0]
    layer1_neurons[1] = 0
    if layer1_neurons[0] == 0:
        layer1_neurons[1] = 1
    layer1_neurons[2] = position[1]
    layer1_neurons[3] = position[2]
    layer1_neurons[4] = position[3]
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


# this will be the computer player's weights
player0_weights = (0.05, 10.0, -50, -0.5, -90, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05)

player_score = [0,0]
player = 0
debug_print(f"Player is {player}")
turn_count = 0
turn_score = 0
turn = 0
pass_turn = 0

colorama_init()
banner()

# play until one or other player has a score of at least 100
while player_score[0] < 100 and player_score[1] < 100:
    time.sleep(1)
    print(f"\nTurn {turn}")
    print(f"Player score 0 is {player_score[0]}, player score 1 is {player_score[1]}")
    turn_count = turn_count + 1
    pass_turn = 0
    if turn_count % 2 == 0:
        turn = turn + 1
        # swap players
    if player == 0:
        print(f"Next player is player 1")
        player = 1
    else:
        print(f"Next player is player 0")
        player = 0
    while pass_turn == 0:
        this_throw = two_dice()
        print(f"Dice throw is {this_throw[0]} and {this_throw[1]}")
        if this_throw[0] == 1 and this_throw[1] == 1:
            print(f"{Fore.RED}oops - two pigs thrown{Style.RESET_ALL}, game score for this player reset to 0 and turn ends")
            turn_score = 0
            player_score[player] = 0
            break
        if this_throw[0] == 1 or this_throw[1] == 1:
            print(f"{Fore.RED}oops - one pig thrown{Style.RESET_ALL}, turn score for this player reset to 0 and turn ends")
            turn_score = 0
            break
        # no pig was thrown so we can bank the score and then decide to continue or not
        turn_score = turn_score + this_throw[0] + this_throw[1]
        # player_score[player] = player_score[player] + turn_score
        if player_score[player] + turn_score >= 100:
            player_score[player] = player_score[player] + turn_score
            print(f"Player {player} has scored {player_score[player]} and has won this game")
            pass_turn = 1
            break
        # time to test the model
        if player == 0:
            position = (turn, player_score[0], player_score[1], turn_score) 
            to_pass = model(position, player0_weights)
        else:
            position = (turn, player_score[1], player_score[0], turn_score)
            to_pass = human(position)
        debug_print(f"to pass score is {to_pass}")
        if to_pass < 50:
            print(f"Player {player} chooses to pass")
            pass_turn = 1
            player_score[player] = player_score[player] + turn_score
            turn_score = 0
        else:
            print(f"Player {player} elects to play, let's roll the dice...")
            
    # report on the end of the games played between the two models (players)
