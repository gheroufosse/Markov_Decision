def markovTuned(
    layout, circle=False, actions=[1, 2, 3], counting=False, itercheck=False
):
    # Version adapted fo empirical tests
    """
  INPUTS : layout, circle, actions, counting
  layout = np.ndarray containing 15 values for the 15 squares
    0 -> ordinary square
    1 -> restart trap
    2 -> penalty trap
    3 -> prison trap
    4 -> bonus
  circle = boolean
    True -> player must land exactly on square 15 to win
    False -> player wins by overstepping the final square
  actions = list (default = [1,2,3])
    list of actions to execute
  counting = bool (default = False)
    set for empirical counting
  itercheck = bool (default = False)
    if True, asks for #iterations before convergence in theoretical case
  ----------------------------
  OUTPUTS : list [Expec,Dice]
  Expec = np.ndarray with the expected cost associated to the 14 squares (goal excluded)
    cost -> number of turns needed to end the game
  Dice = np.ndarray with dice choices for the 14 squares (goal excluded)
    1 -> security dice
    2 -> normal dice
    3 -> risky dice
  """
    # Fast check
    if counting:
        itercheck = False

    # Package
    import numpy as np
    import random

    # Initialization
    states = np.linspace(1, 14, 14, dtype=int)  # Squares (1 -> 14)
    tolerance = 1e-6  # Threshold for VI algorithm
    theo_iter = 0  # Number of iterations before convergence

    # Initializing output arrays
    # Expec : arbitraty choice --> initial values = # steps to reach the goal square
    Expec = np.ndarray(
        (14,), buffer=np.array(np.linspace(14, 1, 14, dtype=float)), dtype=float
    )  # buffer fill the array with data
    Dice = np.ndarray((14,), buffer=np.zeros(14), dtype=int)

    # Will be used to store Expec values before applying the VI algorithm
    old_Expec = np.ndarray((14,), buffer=np.zeros(14), dtype=float)

    ##################
    # Main functions #
    ##################

    def Bellman_iteration(state_i):
        """
    INPUTS : state_i
    state_i = int corresponding to the index of the initial square (1/.../14)
    OUTPUTS : [best_val,best_dice]
    best_val = float corresponding to the min value (#turns) computed for a given state => max_reward ?
    best_dice = int corresponding to the best choice to make at given state
    """

        # List preallocation
        v_values = []
        # Iterate for each action: append computed V value
        for a in actions:
            v_values.append(V_value(state_i, a))
        # Find min V value and corresponding dice
        best_index = v_values.index(min(v_values))
        best_val = v_values[best_index]
        best_dice = actions[best_index]

        return [best_val, best_dice]

    def V_value(state_i, a):
        """
    INPUTS : state_i, a
    state_i = int corresponding to the index of the initial square (1/.../14)
    a = int corresponding to the action that is taken for this computation
    OUTPUT : float corresponding to the computed V value
    """
        # Initialize computation of the V value with the cost of action
        v = 1
        # Find which states are reachable from given initial state and their associated prob
        reach = reachable_states(state_i, a)
        # For each reachable state, compute the right-hand side of the Bellman's eq
        rhs = []
        for s_list in reach:
            state = s_list[0]  # reachable state
            proba = s_list[1]  # proba associated to the new state
            trap_3 = s_list[2]  # bool: prison trap is triggered or not
            bonus = s_list[3]  # bool: bonus is triggered or not
            # Check if goal state is reached
            if state == 15:
                rhs.append(proba * 0)  # V(goal) = 0
            # Check if prison trap has been triggered
            elif trap_3:
                rhs.append(
                    proba * (1 + old_Expec[state - 1])
                )  # add the cost of prison (1 turn)
            # Check if bonus has been triggered
            elif bonus:
                rhs.append(
                    proba * (old_Expec[state - 1] - 1)
                )  # substract 1 turn to the expected cost
            # Regular case
            else:
                rhs.append(proba * old_Expec[state - 1])
        # Sum all values and add to V value
        v += sum(rhs)

        return v

    def steps(dice):
        """
    INPUTS : dice
    dice = dict containing the respective probability OR list containing the choice for each state in the game
    OUTPUT : list of int corresponding to the number of steps needed for each possible start square

    First, choice the reached_state with random weighted choice
    Then, execute the action if trap/bonus is enabled
    Count the turns needed to reach the 15th state
    """

        list_steps = []
        for state in range(1, 15):

            steps = 0

            while state < 15:

                if type(dice) == dict:
                    roll = random.choices(
                        list(dice.keys()), weights=list(dice.values())
                    )
                    reach = reachable_states(state, roll[0])
                elif type(dice) == list:
                    roll = dice[state - 1]
                    reach = reachable_states(state, roll)
                else:
                    wrong = "Invalid dice type"
                    return wrong  # Return error if dice is not a proper list or dico

                steps += 1

                probability = []
                states = []

                for s_list in reach:
                    states.append(s_list[0])
                    probability.append(s_list[1])

                choice = random.choices(reach, weights=probability)[0]
                # print(choice)
                state = choice[0]
                trap_3 = choice[2]  # bool: prison trap is triggered or not
                bonus = choice[3]  # bool: bonus is triggered or not

                if trap_3:
                    steps += 1
                elif bonus:
                    steps -= 1

            list_steps.append(steps)

        return list_steps

    def reachable_states(state_i, action):
        """
    INPUTS : state_i, action
    state_i = int corresponding to the index of the initial square (1/.../14)
    action = int corresponding to the action that is taken for this computation
    OUTPUT : reach
    reach = list of lists giving the reachable states and their associated prob
      content of a nested list:
      [0] int corresponding to the reachable state
      [1] float representing the associated probability of being reached
      [2] bool representing the triggering of trap 3 (prison)
      [3] bool representing the triggering of a bonus (play again)
    """
        # Preallocation
        reach = []
        # Getting all reachable states with given action and from state_i, without effects from traps
        reachable = find_positions(state_i, action)

        # Filling the dictionary

        ##########
        # Case 1 # security dice (0-1) - no trap or bonus
        ##########
        # in both circular and linear boards, there are always 2 or 3 reachable states
        if action == 1:
            # Case without choice between fast or slow lane
            if len(reachable) == 2:  # 2 possibilities because 0 or 1
                for r in reachable:
                    reach.append([r, 0.5, False, False])  # Roll = [0,1]
            # Case with choice between fast and slow lane
            elif len(reachable) == 3:  # 3 possibilities because 0, fast and slow lane
                reach.append([reachable[0], 0.5, False, False])  # Roll = 0
                for r in reachable[1:3]:
                    reach.append([r, 0.25, False, False])  # Roll = [1(fast),1(slow)]

        # Other cases: traps and bonuses may now be triggered
        else:
            # Find new positions due to the effects of traps
            reach_trig = []
            for st in reachable:
                reach_trig.append(triggered(st))

            ##########
            # Case 2 # normal dice (0-1-2) - prob(triggering) = 0.5
            ##########
            # in a circular board: there are 3 or 5 reachable states
            # in a linear board: there are 2, 3 or 5 reachable states

            if action == 2:
                # --------------------------------------------
                # Associating probabilities according to the number of states that are reached
                # 2 states -> 1/3 + 2/3 (2 cases stepped on 15)
                # 3 states -> 3x 1/3
                # 5 states -> 1x 1/3 + 4x 1/6 (2 on fast lane, 2 on slow lane)
                # /!\ : all cases must be divided into 2 situations with 0.5 proba each
                # to represent the triggering of traps and bonuses or not
                # --------------------------------------------
                # Common case: Roll = 0 with proba = 1/3
                reach.append(
                    [reachable[0], 1 / 6, False, False]
                )  # No trap/bonus : 50% of 1/3
                reach.append(
                    [reach_trig[0][0], 1 / 6, reach_trig[0][1], reach_trig[0][2]]
                )  # Trap/bonus : 50% of 1/3
                # 2 states
                if len(reachable) == 2:
                    reach.append([reachable[1], 1 / 3, False, False])  # No trap/bonus
                    reach.append(
                        [reach_trig[1][0], 1 / 3, reach_trig[1][1], reach_trig[1][2]]
                    )
                # 3 states
                elif len(reachable) == 3:
                    reach.append([reachable[1], 1 / 6, False, False])  # No trap/bonus
                    reach.append(
                        [reach_trig[1][0], 1 / 6, reach_trig[1][1], reach_trig[1][2]]
                    )
                    reach.append([reachable[2], 1 / 6, False, False])  # No trap/bonus
                    reach.append(
                        [reach_trig[2][0], 1 / 6, reach_trig[2][1], reach_trig[2][2]]
                    )
                # 5 states
                elif len(reachable) == 5:
                    for i in range(1, 5):
                        reach.append(
                            [reachable[i], 1 / 12, False, False]
                        )  # No trap/bonus
                        reach.append(
                            [
                                reach_trig[i][0],
                                1 / 12,
                                reach_trig[i][1],
                                reach_trig[i][2],
                            ]
                        )

            ##########
            # Case 3 # risky dice (0-1-2-3) - traps and bonuses are always triggered
            ##########
            # in a circular board: there are 4 or 7 reachable states
            # in a linear board: there are 2, 3, 4 or 7 reachable states
            elif action == 3:
                # --------------------------------------------
                # Associating probabilities according to the number of states that are reached
                # 2 states -> 0.25 + 0.75 (3 cases stepped on 15)
                # 3 states -> 0.25 + 0.25 + 0.5 (2 cases stepped on 15)
                # 4 states -> 4x 0.25
                # 7 states -> 1x 0.25 + 6x 0.125 (3 on fast lant, 3 on slow lane)
                # --------------------------------------------
                # Common case: Roll = 0 with proba = 0.25
                reach.append(
                    [reach_trig[0][0], 0.25, reach_trig[0][1], reach_trig[0][2]]
                )
                # 2 states
                if len(reach_trig) == 2:
                    reach.append(
                        [reach_trig[1][0], 0.75, reach_trig[1][1], reach_trig[1][2]]
                    )
                # 3 states
                elif len(reach_trig) == 3:
                    reach.append(
                        [reach_trig[1][0], 0.25, reach_trig[1][1], reach_trig[1][2]]
                    )
                    reach.append(
                        [reach_trig[2][0], 0.50, reach_trig[2][1], reach_trig[2][2]]
                    )
                # 4 states
                elif len(reach_trig) == 4:
                    for i in range(1, 4):
                        reach.append(
                            [reach_trig[i][0], 0.25, reach_trig[i][1], reach_trig[i][2]]
                        )
                # 7 states
                elif len(reach_trig) == 7:
                    for i in range(1, 7):
                        reach.append(
                            [
                                reach_trig[i][0],
                                0.125,
                                reach_trig[i][1],
                                reach_trig[i][2],
                            ]
                        )

        return reach

    def find_positions(state_i, action):
        """
    INPUTS : state_i, action
    state_i = int corresponding to the index of the initial square (1/.../14)
    action = int corresponding to the action that is taken for this computation
    OUTPUT : new_states
    new_states = list of int corresponding to the reachable states when starting
    from state_i with a given action. Traps and bonuses are not taken into account
    """
        # Preallocation
        new_states = []
        # List all special cases for all possible actions: circular board
        cases_circle = {
            1: {3: [4, 11], 10: [15]},
            2: {3: [4, 5, 11, 12], 9: [10, 15], 10: [15, 1], 14: [15, 1]},
            3: {
                3: [4, 5, 6, 11, 12, 13],
                8: [9, 10, 15],
                9: [10, 15, 1],
                10: [15, 1, 2],
                13: [14, 15, 1],
                14: [15, 1, 2],
            },
        }
        # List all special cases for all possible actions: linear board
        cases_linear = {
            1: {3: [4, 11], 10: [15]},
            2: {3: [4, 5, 11, 12], 9: [10, 15], 10: [15], 14: [15]},
            3: {
                3: [4, 5, 6, 11, 12, 13],
                8: [9, 10, 15],
                9: [10, 15],
                10: [15],
                13: [14, 15],
                14: [15],
            },
        }

        # common to all actions
        new_states.append(state_i)  # Roll = 0
        # Get the associated dico of special cases (circular or linear board)
        if circle:
            cases = cases_circle[action]
        elif not circle:
            cases = cases_linear[action]
        # append the output for special cases
        if state_i in cases.keys():
            new_st = cases[state_i]
            for n in new_st:
                new_states.append(n)
        # Regular cases
        else:
            if action == 1:
                new_states.append(state_i + 1)  # Roll = 1
            elif action == 2:
                new_states.append(state_i + 1)  # Roll = 1
                new_states.append(state_i + 2)  # Roll = 2
            elif action == 3:
                new_states.append(state_i + 1)  # Roll = 1
                new_states.append(state_i + 2)  # Roll = 2
                new_states.append(state_i + 3)  # Roll = 3

        return new_states

    def triggered(state):
        """
    INPUT: state
    state = int corresponding to the state that is reached during a play
    OUTPUT: reach_triggered
    reach_triggered = [final_state,prison,again]
        final_state = int corresponding to the final reached state (effect of traps)
        prison = bool representing the triggering of trap 3 (prison)
        again = bool representing the triggering of a bonus (play again)
    """
        # Dico used for special cases when triggering trap 2 (moving backward)
        trap_2 = {1: 1, 2: 1, 3: 1, 11: 1, 12: 2, 13: 3}
        # Find the trap/bonus and apply its effect
        trig = layout[state - 1]
        if trig == 0:  # No trap or bonus
            reach_triggered = [state, False, False]
        elif trig == 1:  # Teleportation trap
            reach_triggered = [1, False, False]
        elif trig == 2:  # Backward trap
            if state in trap_2.keys():
                reach_triggered = [trap_2[state], False, False]
            else:
                reach_triggered = [state - 3, False, False]
        elif trig == 3:  # Prison trap
            reach_triggered = [state, True, False]
        elif trig == 4:  # Bonus
            reach_triggered = [state, False, True]

        return reach_triggered

    ################
    # VI algorithm #
    ################

    if counting:
        # Count the number of steps to reach the 15th state
        return steps(actions)

    else:
        # Convergence condition defined by the chosen tolerance
        while max(abs(Expec - old_Expec)) > tolerance:

            # Update number of iterations
            theo_iter += 1
            # Store previous values in Expec before starting a new iteration
            old_Expec = Expec.copy()
            # Loop on all states
            for s in range(len(states)):
                state = states[s]
                # Compute best value and best policy starting from state s
                [best_val, best_dice] = Bellman_iteration(state)
                # Update Expec with best computed value (min cost) starting from state s
                Expec[s] = best_val
                # Update Dice with best choice starting from state s
                Dice[s] = best_dice

        if itercheck:
            return [Expec, Dice, theo_iter]
        else:
            return [Expec, Dice]


#%%

# Random generation of layouts

import random
import numpy as np


def generator_layout(nb_restart, nb_penalty, nb_prison, nb_bonus):
    """
    INPUTS : nb_restart, nb_penalty, nb_prison, nb_bonus
    nb_restart = int corresponding to the number of restart traps in the layout
    nb_penalty = int corresponding to the number of penalty traps in the layout
    nb_prison = int corresponding to the number of prison traps in the layout
    nb_bonus = int corresponding to the number of bonuses in the layout
    --------------------------------------------------------------
    OUTPUTS : array corresponding to the layout
    """

    lay = np.zeros(15, dtype=int)
    # Assume tap 1 and 15 are 0
    if (nb_bonus + nb_penalty + nb_prison + nb_restart) > 13:
        print("Error : input number")
        return lay
    lay = fill_layout(lay, 1, nb_restart)
    lay = fill_layout(lay, 2, nb_penalty)
    lay = fill_layout(lay, 3, nb_prison)
    lay = fill_layout(lay, 4, nb_bonus)
    return lay


def fill_layout(lay, trap, iteration):
    """
    INPUTS : lay, trap, iteration
    lay = array corresponding to the current layout
    trap = int corresponding to the type of traps to include
    iteration = int corresponding to the number of traps to include
    --------------------------------------------------------------
    OUTPUTS : array corresponding to the new layout
    """

    # random.seed(0) Add this line to reobtain the same layout
    for i in range(iteration):
        rand = random.randint(1, 13)
        # while a number has already be placed, find another empty place (with 0)
        while lay[rand] != 0:
            rand = random.randint(1, 13)
        lay[rand] = trap
    return lay


#%%

# Empirical experiment to validate the Snake-game Markov Decision Process

import time


def create_player():
    """Function to create the differents players types

    Returns:
        player1, player2, player3 : dictionnaries with probabilities associated to each dice according to risk aversion
    """
    random.seed(0)
    PLAYERS = {}
    PROBA_J1 = random.uniform(0, 0.5)
    PROBA_J3 = random.uniform(0, 0.5)

    PROBAS = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0.5, PROBA_J1, 0.5 - PROBA_J1],
        [1 / 3] * 3,
        [PROBA_J3, 0.5 - PROBA_J3, 0.5],
    ]

    for a in range(len(PROBAS)):
        PLAYERS[f"player{a+1}"] = {}
        for i in range(3):
            PLAYERS[f"player{a+1}"][i + 1] = PROBAS[a][i]
        time.sleep(1)
        print("Generating Player", end=" ")
        for i in range(3):
            time.sleep(0.15)
            print(".", end=" ")
        time.sleep(0.3)
        print(f"\nPlayer {a+1}: {PLAYERS[f'player{a+1}']}")

    return PLAYERS


def Counting(dices, layout, n=10000, circle=False):
    """Count the number of turns needed to reach the last square for 10.000 simulations

    Arguments:
        dices {dictionnary/list} -- List or dictionary for the movements
        n {int} -- Number of iterations to run (Default = 10000)
        layout {np.array} -- Layout used for the simulation

    Returns:
        dict of dicts, for each initial state
        Dict -- Index : Turns taken ; Value : Count the number of times it took [index] turns over the n iterations
    """
    list_count = {}
    for i in range(14):
        list_count[i + 1] = {}

    for i in range(n):

        steps = markovTuned(layout, circle, actions=dices, counting=True)

        for j in range(len(steps)):
            if steps[j] in list_count[j + 1].keys():
                list_count[j + 1][steps[j]] += 1
            else:
                list_count[j + 1][steps[j]] = 1

    return list_count


def get_results(layout, circle=False):
    """Function that returns the results of 10.000 empirical simulations of the Snake and Ladders game

    Keyword Arguments:
        layout {np.array} -- Layout used for the empirical analysis

    Returns:
        dict -- dictionnary with for key players and their associated results
    """

    Expected_Turns, COMPUTER_DICES = markovTuned(layout, circle)
    print("Optimal strategy : ", COMPUTER_DICES)
    PLAYERS = create_player()

    results = {}
    for player, dices in PLAYERS.items():
        print("Start computing - {0}".format(player))
        results[player] = Counting(dices, layout, 10000, circle)

    print("Start computing with optimal strategy")
    results["computer"] = Counting(list(COMPUTER_DICES), layout, 10000, circle)

    return results


#%%

# Check the computed expected cost for the optimal strategy and compare with theoretical results

results = get_results(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), True)

import math

for i in range(1, 15):
    a = results["computer"][i]
    iters = 0
    summation = 0
    somme = 0
    std = 0
    for key, val in a.items():
        summation += key * val
        iters += val
    mean = summation / iters
    for key, val in a.items():
        somme += val * (key - mean) ** 2
    std = math.sqrt(somme / iters)
    print(f"Case {i} mean: {mean} turns")
    print(f"Case {i} std: {std}")

# Theoretical results
markovTuned(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), True)[0]


#%%

# Statistical analysis for a specific layout

import matplotlib.pyplot as plt
import statistics

plt.style.use("ggplot")

dico = {}
for key, val in results.items():

    dico[key] = []
    # print(val[1])
    for i, j in val[1].items():
        for k in range(j):
            dico[key].append(i)

means = []
medians = []
for k, v in dico.items():
    means.append(sum(v) / len(v))
    medians.append(statistics.median(v))


fig, ax = plt.subplots(figsize=(10, 6))

# Add a smooth grid behind the plot
ax.yaxis.grid(True, linestyle="-", which="major", color="black", alpha=0.16)
ax.set(axisbelow=True)  # Hide the grid behind plot objects

plt.ylabel("Turns needed to reach last square", fontsize=14)
ax.boxplot(dico.values(), notch=0, sym="+", vert=1, whis=1.5)


for i in range(len(means)):
    ax.plot(i + 1, means[i], color="r", marker="*", markeredgecolor="r")
    ax.text(
        i + 1,
        0.95,
        means[i],
        transform=ax.get_xaxis_transform(),
        horizontalalignment="center",
        size="small",
        color="r",
        weight="bold",
    )

# Set the axes ranges and axes labels
top = 100
bottom = 0
ax.set_ylim(bottom, top)
ax.set_xticklabels(dico.keys(), rotation=45, fontsize=10)

fig.text(
    0.77, 0.005, "*", color="r", backgroundcolor="silver", weight="roman", size="medium"
)
fig.text(0.785, 0, " Mean value", color="black", weight="roman", size="small")

plt.tight_layout
plt.savefig("output_1.png")
plt.show()
