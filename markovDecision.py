def markovDecision(layout,circle):
  """
                                        ___                             
                                        (   )                            
    ___ .-. .-.     .---.   ___ .-.     | |   ___     .--.    ___  ___  
   (   )   '   \   / .-, \ (   )   \    | |  (   )   /    \  (   )(   ) 
    |  .-.  .-. ; (__) ; |  | ' .-. ;   | |  ' /    |  .-. ;  | |  | |  
    | |  | |  | |   .'`  |  |  / (___)  | |,' /     | |  | |  | |  | |  
    | |  | |  | |  / .'| |  | |         | .  '.     | |  | |  | |  | |  
    | |  | |  | | | /  | |  | |         | | `. \    | |  | |  | |  | |  
    | |  | |  | | ; |  ; |  | |         | |   \ \   | '  | |  ' '  ; '  
    | |  | |  | | ' `-'  |  | |         | |    \ .  '  `-' /   \ `' /   
   (___)(___)(___)`.__.'_. (___)       (___ ) (___)  `.__.'     '_.'    
                                                                        
  INPUTS : layout, circle
  layout = np.ndarray containing 15 values for the 15 squares
    0 -> ordinary square
    1 -> restart trap
    2 -> penalty trap
    3 -> prison trap
    4 -> bonus
  circle = boolean
    True -> player must land exactly on square 15 to win
    False -> player wins by overstepping the final square
  ----------------------------
  OUTPUTS : list [Expec,Dice]
  Expec = np.ndarray with the expected cost associated to the 14 squares (goal excluded)
    cost -> number of turns needed to end the game
  Dice = np.ndarray with dice choices for the 14 squares (goal excluded)
    1 -> security dice
    2 -> normal dice
    3 -> risky dice
  """
  # Package
  import numpy as np
  # Initialization
  actions = [1,2,3]                         # Dice
  states = np.linspace(1,14,14,dtype=int)   # Squares (1->14)
  tolerance = 1e-6                          # Threshold for VI algorithm
  # Initializing output arrays
  # Expec : arbitraty choice --> initial values = #steps to reach the goal square
  Expec = np.ndarray((14,), buffer=np.array(np.linspace(14,1,14,dtype=float)), dtype=float)
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
    best_val = float corresponding to the min value (#turns) computed for a given state
    best_dice = int corresponding to the best choice to make at given state
    """
    # List preallocation
    v_values = []
    # Iterate for each action: append computed V value
    for a in actions:
      v_values.append(V_value(state_i,a))
    # Find min V value and corresponding dice
    best_index = v_values.index(min(v_values))
    best_val = v_values[best_index]
    best_dice = actions[best_index]

    return [best_val,best_dice]

  def V_value(state_i,a):
    """
    INPUTS : state_i, a
    state_i = int corresponding to the index of the initial square (1/.../14)
    a = int corresponding to the action that is taken for this computation
    OUTPUT : float corresponding to the computed V value
    """
    # Initialize computation of the V value with the cost of action
    v = 1
    # Find which states are reachable from given initial state and their associated prob
    reach = reachable_states(state_i,a)
    # For each reachable state, compute the right-hand side of the Bellman's eq
    rhs = []
    for s_list in reach:
      state = s_list[0]       # reachable state
      proba = s_list[1]       # proba associated to the new state
      trap_3 = s_list[2]      # bool: prison trap is triggered or not
      bonus = s_list[3]       # bool: bonus is triggered or not
      # Check if goal state is reached
      if state == 15:
        rhs.append(proba*0)   # V(goal) = 0
      # Check if prison trap has been triggered
      elif trap_3:
        rhs.append(proba*(1 + old_Expec[state-1]))  # add the cost of prison (1 turn)
      # Check if bonus has been triggered
      elif bonus:
        rhs.append(proba*(old_Expec[state-1] - 1))  # substract 1 turn to the expected cost
      # Regular case
      else:
        rhs.append(proba*old_Expec[state-1])
    # Sum all values and add to V value
    v += sum(rhs)

    return v

  def reachable_states(state_i,action):
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
    # Getting all reachable states with given action and from state_i
    reachable = find_positions(state_i,action)

    # Filling the dictionary

    ##########
    # Case 1 # security dice (0-1) - no trap or bonus
    ##########
    # in both circular and linear boards, there are always 2 or 3 reachable states
    if action == 1:
      # No choice between fast or slow lane
      if len(reachable) == 2:
        for r in reachable:
          reach.append([r,0.5,False,False])           # Roll = [0,1]
      # Choice between fast and slow lane
      elif len(reachable) == 3:
        reach.append([reachable[0],0.5,False,False])  # Roll = 0
        for r in reachable[1:3]:
          reach.append([r,0.25,False,False])          # Roll = [1(fast),1(slow)]

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
        #--------------------------------------------
        # Associating probabilities according to the number of states that are reached
        # 2 states -> 1/3 + 2/3 (2 cases stepped on 15)
        # 3 states -> 3x 1/3 
        # 5 states -> 1x 1/3 + 4x 1/6 (2 on fast lane, 2 on slow lane)
        # /!\ : all cases must be divided into 2 situations with 0.5 proba each
        # to represent the triggering of traps and bonuses or not
        #--------------------------------------------
        # Common case: Roll = 0 with proba = 1/3
        reach.append([reachable[0], 1/6, False, False])       # No trap/bonus
        reach.append([reach_trig[0][0], 1/6, reach_trig[0][1], reach_trig[0][2]])
        # 2 states
        if len(reachable) == 2:
          reach.append([reachable[1], 1/3, False, False])     # No trap/bonus
          reach.append([reach_trig[1][0], 1/3, reach_trig[1][1], reach_trig[1][2]])
        # 3 states
        elif len(reachable) == 3:
          reach.append([reachable[1], 1/6, False, False])     # No trap/bonus
          reach.append([reach_trig[1][0], 1/6, reach_trig[1][1], reach_trig[1][2]])
          reach.append([reachable[2], 1/6, False, False])     # No trap/bonus
          reach.append([reach_trig[2][0], 1/6, reach_trig[2][1], reach_trig[2][2]])
        # 5 states
        elif len(reachable) == 5:
          for i in range(1,5):
            reach.append([reachable[i], 1/12, False, False])  # No trap/bonus
            reach.append([reach_trig[i][0], 1/12, reach_trig[i][1], reach_trig[i][2]])

      ##########
      # Case 3 # risky dice (0-1-2-3) - traps and bonuses are always triggered
      ##########
      # in a circular board: there are 4 or 7 reachable states
      # in a linear board: there are 2, 3, 4 or 7 reachable states
      elif action == 3:
        #--------------------------------------------
        # Associating probabilities according to the number of states that are reached
        # 2 states -> 0.25 + 0.75 (3 cases stepped on 15)
        # 3 states -> 0.25 + 0.25 + 0.5 (2 cases stepped on 15)
        # 4 states -> 4x 0.25
        # 7 states -> 1x 0.25 + 6x 0.125 (3 on fast lant, 3 on slow lane)
        #--------------------------------------------
        # Common case: Roll = 0 with proba = 0.25
        reach.append([reach_trig[0][0], 0.25, reach_trig[0][1], reach_trig[0][2]])
        # 2 states
        if len(reach_trig) == 2:
          reach.append([reach_trig[1][0], 0.75, reach_trig[1][1], reach_trig[1][2]])
        # 3 states
        elif len(reach_trig) == 3:
          reach.append([reach_trig[1][0], 0.25, reach_trig[1][1], reach_trig[1][2]])
          reach.append([reach_trig[2][0], 0.50, reach_trig[2][1], reach_trig[2][2]])
        # 4 states
        elif len(reach_trig) == 4:
          for i in range(1,4):
            reach.append([reach_trig[i][0], 0.25, reach_trig[i][1], reach_trig[i][2]])
        # 7 states
        elif len(reach_trig) == 7:
          for i in range(1,7):
            reach.append([reach_trig[i][0], 0.125, reach_trig[i][1], reach_trig[i][2]])

    return reach

  def find_positions(state_i,action):
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
    cases_circle = {1:{3:[4,11], 10:[15]},
                    2:{3:[4,5,11,12], 9:[10,15], 10:[15,1], 14:[15,1]},
                    3:{3:[4,5,6,11,12,13], 8:[9,10,15],
                       9:[10,15,1], 10:[15,1,2], 13:[14,15,1], 14:[15,1,2]}}
    # List all special cases for all possible actions: linear board
    cases_linear = {1:{3:[4,11], 10:[15]},
                    2:{3:[4,5,11,12], 9:[10,15], 10:[15], 14:[15]},
                    3:{3:[4,5,6,11,12,13], 8:[9,10,15],
                       9:[10,15], 10:[15], 13:[14,15], 14:[15]}}

    # common to all actions
    new_states.append(state_i)        # Roll = 0
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
        new_states.append(state_i+1)  # Roll = 1
      elif action == 2:
        new_states.append(state_i+1)  # Roll = 1
        new_states.append(state_i+2)  # Roll = 2
      elif action == 3:
        new_states.append(state_i+1)  # Roll = 1
        new_states.append(state_i+2)  # Roll = 2
        new_states.append(state_i+3)  # Roll = 3

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
    trap_2 = {1:1,2:1,3:1,11:1,12:2,13:3}
    # Find the trap/bonus and apply its effect
    trig = layout[state-1]
    if trig == 0:     # No trap or bonus
      reach_triggered = [state,False,False]
    elif trig == 1:   # Teleportation trap
      reach_triggered = [1,False,False]
    elif trig == 2:   # Backward trap
      if state in trap_2.keys():
        reach_triggered = [trap_2[state],False,False]
      else:
        reach_triggered = [state-3,False,False]
    elif trig == 3:   # Prison trap
      reach_triggered = [state,True,False]
    elif trig == 4:   # Bonus
      reach_triggered = [state,False,True]

    return reach_triggered

  ################
  # VI algorithm #
  ################
  
  # Convergence condition defined by the chosen tolerance
  while max(abs(Expec - old_Expec)) > tolerance:

    # Store previous values in Expec before starting a new iteration
    old_Expec = Expec.copy()
    # Loop on all states
    for s in range(len(states)):
      state = states[s]
      # Compute best value and best policy starting from state s
      [best_val,best_dice] = Bellman_iteration(state)
      # Update Expec with best computed value (min cost) starting from state s
      Expec[s] = best_val
      # Update Dice with best choice starting from state s
      Dice[s] = best_dice
    
  return [Expec,Dice]