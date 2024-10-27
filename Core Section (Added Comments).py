def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable):
    # Loop over the number of training episodes
    for episode in trange(n_training_episodes):

        # Dynamically calculate epsilon for the current episode, decaying it over time to reduce exploration
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

        # Reset the environment to the initial state at the start of each episode
        state = env.reset()
        step = 0
        done = False

        # Loop for the maximum number of steps in each episode
        for step in range(max_steps):

            # Select an action using the epsilon-greedy policy
            action = epsilon_greedy_policy(Qtable, state, epsilon)

            # Take the selected action, observe the new state and reward from the environment
            new_state, reward, done, info = env.step(action)

            # Update the Q-value for the state-action pair using the Bellman equation
            # Q(s, a) = Q(s, a) + learning_rate * [reward + gamma * max(Q(s', a')) - Q(s, a)]
            Qtable[state][action] = Qtable[state][action] + learning_rate * (reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action])

            # If the episode ends (goal reached or falls into a hole), break the loop
            if done:
                break

            # Update the current state to the new state for the next step
            state = new_state
    
    # Return the updated Q-table after completing all episodes
    return Qtable