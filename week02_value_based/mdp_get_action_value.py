
def get_action_value(mdp, state_values, state, action, gamma):
    """ Computes Q(s,a) as in formula above """

    q = 0
    next_states = mdp.get_next_states(state, action)
    for s, p in next_states.items():
        q += p * (mdp.get_reward(state, action, s) + gamma * state_values[s])
        
    return q