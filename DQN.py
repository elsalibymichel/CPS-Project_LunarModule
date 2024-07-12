import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from collections import deque, namedtuple
from keras.optimizers import Adam
from keras.losses import MSE


# Function that updates the target network weights
def update_networks(q_network, target_q_network, tau):
    for target_weights, q_net_weights in zip(
            target_q_network.weights, q_network.weights
    ):
        target_weights.assign(tau * q_net_weights + (1.0 - tau) * target_weights)


# Define the loss function
def compute_loss(experiences, gamma, q_network, target_q):
    states, actions, rewards, next_states, done_vals = experiences

    # Get the maximum q value of the target q network
    max_qsa = tf.reduce_max(target_q(next_states), axis=-1)
    # Compute the target q values
    y_targets = rewards + (gamma * max_qsa) * (1 - done_vals)
    # Get the q values of the q network
    q_values = q_network(states)
    # Reorder the q values
    q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]),
                                                tf.cast(actions, tf.int32)], axis=1))
    # Compute the mean squared error
    loss = MSE(y_targets, q_values)

    return loss


class DQN:

    def __init__(self, learning_rate, size_batch=None, tau=None, epsilon_dec=None, epsilon_min=None, seed=None):

        # Batch dimension
        if size_batch is None:
            self.size_batch = 64
        else:
            self.size_batch = size_batch

        # Soft update parameter
        if tau is None:
            self.tau = 1e-3
        else:
            self.tau = tau

        # Epsilon-decay rate
        if epsilon_dec is None:
            self.epsilon_dec = 0.995
        else:
            self.epsilon_dec = epsilon_dec

        # Minimum epsilon value
        if epsilon_min is None:
            self.epsilon_min = 0.01
        else:
            self.epsilon_min = epsilon_min

        # Optionally, set the random seed for reproducibility
        if seed is not None:
            random.seed(seed)

        # Initial epsilon value
        self.epsilon = 1

        self.optimizer = Adam(learning_rate=learning_rate)

        # Define the agent learning function

    @tf.function
    def agent_learn(self, experiences, gamma, q_network, target_q, tau):
        # Compute the loss function
        with tf.GradientTape() as tape:
            loss = compute_loss(experiences, gamma, q_network, target_q)

            # Define the gradients
        gradients = tape.gradient(loss, q_network.trainable_variables)
        # Apply the gradients
        self.optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))
        # Update the weights of both networks
        update_networks(q_network, target_q, tau)

    # Function that returns the action to take with epsilon-greedy policy
    def get_action(self, q_values):
        # With probability epsilon, take a random action
        if random.random() > self.epsilon:
            return np.argmax(q_values.numpy()[0])
        else:
            return random.choice(np.arange(4))

    def calculate_reward(self, state, action, done, prev_shaping):
        pos_x, pos_y, vel_x, vel_y, angle, angular_vel, leg_contact_left, leg_contact_right = state

        # Inizialize reward
        reward = 0

        # Dense Reward
        shaping = (
                - 140 * np.sqrt(pos_x ** 2 + pos_y ** 2)  # Distance from the landing platform
                - 100 * np.sqrt(vel_x ** 2 + vel_y ** 2)  # Difference from the references velocity
                - 120 * abs(angle)  # Difference from the references angular
                + 10 * leg_contact_left  # Reward if the left leg hit the ground
                + 10 * leg_contact_right  # Reward if the right leg hit the ground
        )

        # Check for the previous shaping
        if prev_shaping is not None:
            reward = shaping - prev_shaping

        # Main engine
        if action == 2:
            reward += 0.3  # less fuel spent is better

        # Left engine and Right engine
        if action == 1 or action == 3:
            reward += 0.03  # less fuel spent is better

        # Sparse reward for a good or bad landing
        if done:
            if not (leg_contact_left and leg_contact_right):
                reward = -100
            else:
                reward = +100

        return reward, shaping

    # Function that return the initial conditions
    def choose_initial_condition(self):

        # Sample a random initial condition
        initial_condition = np.array([random.uniform(-0.1, 0.1), random.uniform(1.39, 1.41), random.uniform(-1, 1),
                                      random.uniform(-0.7, 0.7), random.uniform(-0.4, 0.4), 0.0, 0.0, 0.0])

        return initial_condition

    # Function that returns a batch of experiences from the replay buffer
    def get_experiences(self, memory_buffer):

        # Sample a batch of experiences from the replay buffer
        experiences = random.sample(memory_buffer, k=self.size_batch)

        # Convert the states of the experiences to tensors
        states = tf.convert_to_tensor(
            np.array([e.state for e in experiences if e is not None]), dtype=tf.float32
        )

        # Convert the actions of the experiences to tensors
        actions = tf.convert_to_tensor(
            np.array([e.action for e in experiences if e is not None]), dtype=tf.float32
        )

        # Convert the rewards of the experiences to tensors
        rewards = tf.convert_to_tensor(
            np.array([e.reward for e in experiences if e is not None]), dtype=tf.float32
        )

        # Convert the next states of the experiences to tensors
        next_states = tf.convert_to_tensor(
            np.array([e.next_state for e in experiences if e is not None]), dtype=tf.float32
        )

        # Convert the done values of the experiences to tensors
        done_vals = tf.convert_to_tensor(
            np.array([e.done for e in experiences if e is not None]).astype(np.uint8),
            dtype=tf.float32,
        )

        return states, actions, rewards, next_states, done_vals

    # Function that checks if the number of steps is a multiple of the number of steps to update
    def check_update_conditions(self, t, num_steps_upd, memory_buffer):
        if (t + 1) % num_steps_upd == 0 and len(memory_buffer) > self.size_batch:
            return True
        else:
            return False

    # Function that update the epsilon value with the decay rate
    def new_eps(self):
        self.epsilon = max(self.epsilon_min, self.epsilon_dec * self.epsilon)

    # Function that performs the DQN algorithm given the environment, the Q-network, the target Q-network,
    # the agent learning function, the steps function, the size of the memory buffer, the discount factor gamma,
    # the number of steps to update the network, the maximum number of episodes, and the maximum number of steps
    def DQN(self, env, q_network, target_q, early_stop=None,
            size_memory=100000, gamma=0.995, steps_update=4, max_episodes=2000, max_steps=1000):

        # Initialize the epsilon value
        self.epsilon = 1

        # Create a namedtuple for the experiences
        memory_tuple = namedtuple("Steps", field_names=["state", "action", "reward", "next_state", "done"])

        # Create a replay buffer
        memory_buffer = deque(maxlen=size_memory)

        # Set the target network weights to be the same as the Q-network weights
        target_q.set_weights(q_network.get_weights())

        # Create lists to store the cumulative discount rewards and the mean cumulative discount rewards
        cDR_list = list()
        cDR_mean_list = list()

        # Loop over the episodes
        for ii in range(max_episodes):

            # Reset the environment
            env.reset()
            env.x0 = self.choose_initial_condition()
            state = env.x0.copy()

            # Initialize the cumulative discount reward
            cDR = 0

            # Initialize the prev_shaping
            prev_shaping = None

            # Loop over the steps
            for t in range(max_steps):

                # Convert the state to have the proper dimensions
                state_qn = np.expand_dims(state, axis=0)
                # Get the Q-values associated to the current state
                q_values = q_network(state_qn)
                # Get the action to perform with epsilon-greedy policy
                action = self.get_action(q_values)

                # Perform the selected action
                next_state, _, done, _, info = env.step(action)

                # Compute the reward
                reward, shaping = self.calculate_reward(next_state, action, done, prev_shaping)

                # Compute the cumulative discount reward
                cDR += (gamma ** t) * reward

                # Save the experience in the replay buffer
                memory_buffer.append(memory_tuple(state, action, reward, next_state, done))

                # Update prev_shaping for the next step
                prev_shaping = shaping

                # Check if the number of steps is a multiple of the number of steps to update
                update = self.check_update_conditions(t, steps_update, memory_buffer)

                if update:
                    # Get a random batch of experiences from the replay buffer
                    experiences = self.get_experiences(memory_buffer)
                    # Update the Q-network weights
                    self.agent_learn(experiences, gamma, q_network, target_q, self.tau)

                # Update the state
                state = next_state.copy()

                # Verify if the episode is terminated
                if done:
                    break

            # Update epsilon
            self.new_eps()

            # Save progress
            cDR_list.append(cDR)
            cDR_mean = np.mean(cDR_list[-100:])
            cDR_mean_list.append(cDR_mean)

            # Display progress
            print(f"\rEpisode {ii + 1} | CDR: {cDR}", end="")
            if (ii + 1) % 100 == 0:
                print(f"\rEpisode {ii + 1} | Average CDR over the last 100 episodes: {cDR_mean}")

            # Stop simulation if the mean cumulative discount reward is greater than 110
            if early_stop is not None:
                if cDR_mean >= early_stop:
                    print(f"\n\nSolution found in {ii + 1} episodes!")
                    break

        return cDR_list, cDR_mean_list


# Function to plot the history of the rewards
def plot_history(points_history, points_mean_history):
    points = points_history[:len(points_history)]
    points_mean = points_mean_history[:len(points_mean_history)]

    episode_num = [x for x in range(len(points_history))]

    plt.plot(episode_num, points, linewidth=1, label="Reward")
    plt.plot(episode_num[99:], points_mean[99:], linewidth=2, label="Avg Reward")

    plt.grid()
    plt.title("Reward History")
    plt.xlabel("Episode")
    plt.ylabel("Total Points")
    plt.legend()
    plt.show()