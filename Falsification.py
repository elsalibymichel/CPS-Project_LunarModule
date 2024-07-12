import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
from moonlight import *
from matplotlib import animation
import random
import math
import pandas as pd


# Extract moonlight results in a list
def extract(ls, i):
    return list(map(lambda x: x[i], ls))


# Return the indices of the n individual that have the minimum values of robustness
# and returns also the minimum value of the robustness list
def get_best_n_individual(robustness, n):
    index_best_individual = list()
    min_robustness = sorted(robustness)[:n]
    minimum_robustness = sorted(robustness)[0]
    for value in min_robustness:
        index_best_individual.append(robustness.index(value))
    return index_best_individual, minimum_robustness


# Save all frames of a set of simulations in a .gif
def save_frames_as_gif(frames, name):
    filename = f'Falsification_{name}.gif'
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save('./' + filename, writer='imagemagick', fps=60)


class MoonlightFalsification:

    def __init__(self, model_path, constraint_x=None, threshold_theta=None, constraint_theta=None,
                 constraint_speed=None, threshold_land_x=None, threshold_speed=None, constraint_land_y=None,
                 constraint_land_x=None, admissible_initial_conditions=None):

        self.counts_falsification = None
        self.falsification_iterations = None
        self.falsification_list = None
        self.best_individual = None

        self.n_parents = None
        self.n_offspring_per_parent = None
        self.n_pop = None
        self.mutation_rate = None
        self.max_offspring_retries = None
        self.simulation_steps = None

        self.exp_name = None

        # X constraint
        if constraint_x is None:
            self.constraint_x = 0.8
        else:
            self.constraint_x = constraint_x

        # Theta threshold
        if threshold_theta is None:
            self.threshold_theta = 0.4
        else:
            self.threshold_theta = threshold_theta

        # Theta constraint
        if constraint_theta is None:
            self.constraint_theta = 0.6
        else:
            self.constraint_theta = constraint_theta

        # Speed constraint
        if constraint_speed is None:
            self.constraint_speed = 1
        else:
            self.constraint_speed = constraint_speed

        # Landing x threshold
        if threshold_land_x is None:
            self.threshold_land_x = 0.1
        else:
            self.threshold_land_x = threshold_land_x

        # Speed threshold
        if threshold_speed is None:
            self.threshold_speed = 0.4
        else:
            self.threshold_speed = threshold_speed

        # Landing x constraint
        if constraint_land_x is None:
            self.constraint_land_x = 0.25
        else:
            self.constraint_land_x = constraint_land_x

        # Landing y constraint
        if constraint_land_y is None:
            self.constraint_land_y = 0.1
        else:
            self.constraint_land_y = constraint_land_y

        # Admissible initial condition for the simulation
        if admissible_initial_conditions is None:
            self.admissible_initial_conditions = [[-0.05, 0.05], [1.39, 1.41], [-0.5, 0.5], [-0.3, 0], [-0.1, 0.1]]
        else:
            self.admissible_initial_conditions = admissible_initial_conditions

        # Import the model
        self.model = load_model(model_path)

        # Creation the Moonlight Script for the formulae
        self.MoonlightScript = self.verification()

        # Collect the constraints
        self.normalization_factors = [self.constraint_x, self.constraint_theta, self.constraint_speed,
                                      self.constraint_land_x, self.constraint_land_y]

    # Function that mutate the initial conditions of a list of parents and return a list of mutated offspring
    def get_offspring(self, index_parents, initial_states):
        offspring = list()

        for index in index_parents:
            for ii in range(self.n_parents):
                admissible = False
                retries = 1
                offspring_proposal = None

                # When generating the offspring, we need to make sure that the initial conditions are admissible
                while not admissible and retries <= self.max_offspring_retries:

                    offspring_proposal = [initial_states[index][kk] + np.random.normal(
                        0, self.mutation_rate * (
                                self.admissible_initial_conditions[kk][1] - self.admissible_initial_conditions[kk][0])
                    ) for kk in range(5)] + [0.0, 0.0, 0.0]

                    for jj in range(5):
                        if not (self.admissible_initial_conditions[jj][0] <= offspring_proposal[jj] <=
                                self.admissible_initial_conditions[jj][1]):
                            admissible = False
                            break
                        else:
                            admissible = True
                    retries += 1

                # If after max_offspring_retries the offspring is not admissible, we choose a new initial condition
                if not admissible:
                    offspring_proposal = self.choose_initial_condition()

                offspring.append(offspring_proposal)

        return offspring

    # Function that return the initial conditions
    def choose_initial_condition(self):
        # Sample a random initial condition
        initial_condition = np.array([random.uniform(self.admissible_initial_conditions[0][0],
                                                     self.admissible_initial_conditions[0][1]),
                                      random.uniform(self.admissible_initial_conditions[1][0],
                                                     self.admissible_initial_conditions[1][1]),
                                      random.uniform(self.admissible_initial_conditions[2][0],
                                                     self.admissible_initial_conditions[2][1]),
                                      random.uniform(self.admissible_initial_conditions[3][0],
                                                     self.admissible_initial_conditions[3][1]),
                                      random.uniform(self.admissible_initial_conditions[4][0],
                                                     self.admissible_initial_conditions[4][1]),
                                      0.0, 0.0, 0.0]  # The last three states are not modified
                                     )
        return initial_condition

    # Function that given the robustness of a generation of individuals, return the next generation
    def genetic_algorith(self, robustness, initial_states):

        # Search the index minimum robustness
        index_parents, _ = get_best_n_individual(robustness, self.n_offspring_per_parent)

        # Generate the offspring
        initial_states = self.get_offspring(index_parents, initial_states)

        return initial_states

    # Define the formulae for falsification
    def verification(self):

        # Script for the robustness
        script = f'''    
        signal {{real abs_x; real abs_theta; real vel_y; real y; }}
        domain minmax;
        formula general_x = globally ( abs_x < {self.constraint_x} );
        formula rotation = globally ( ( y <= {self.threshold_theta} ) -> ( abs_theta < {self.constraint_theta}) );
        formula velocity = globally ( ( y <= {self.threshold_speed} ) -> ( vel_y < {self.constraint_speed}) );
        formula land_x = globally (( y <= {self.threshold_land_x} ) ->  (abs_x < {self.constraint_land_x}) );
        formula land_y = (true) until (globally (y <= {self.constraint_land_y}));
        '''

        # Load the formulae on Moonlight
        moonlightScript = ScriptLoader.loadFromText(script)

        return moonlightScript

    # Function that computes the robustness and checks
    # if the specifications are True or False
    def get_robustness(self, trajectories, initial_states, frames):

        # Initialize the robustness list and done variable
        robustness = list()
        # done = False

        # Verify all the n_pop trajectories
        for ii in range(self.n_pop):

            # Define time
            time = np.arange(0, len(trajectories[ii][:, 0]), 1.).tolist()

            # Define formula for x
            monitor_general_x = self.MoonlightScript.getMonitor("general_x")
            # Define formula for theta
            monitor_rotation = self.MoonlightScript.getMonitor("rotation")
            # Define formula for velocity
            monitor_velocity = self.MoonlightScript.getMonitor("velocity")
            # Define formula for landing x
            monitor_land_x = self.MoonlightScript.getMonitor("land_x")
            # Define formula for landing y
            monitor_land_y = self.MoonlightScript.getMonitor("land_y")

            # Adapt the trajectories data-structure to fit the Moonlight script
            values = list(zip(np.float64(np.abs(np.squeeze(trajectories[ii][:, 0]))),
                              np.float64(np.squeeze(np.abs(trajectories[ii][:, 4]))),
                              np.float64(np.abs(np.squeeze(trajectories[ii][:, 3]))),
                              np.float64(np.squeeze(trajectories[ii][:, 1]))))

            # Get results of the formulas
            result_general_x = monitor_general_x.monitor(time, values)
            result_rotation = monitor_rotation.monitor(time, values)
            result_velocity = monitor_velocity.monitor(time, values)
            result_land_x = monitor_land_x.monitor(time, values)
            result_land_y = monitor_land_y.monitor(time, values)

            # Extract results x and time
            plot_general_x = extract(result_general_x, 1)
            plot_time_general_x = extract(result_general_x, 0)
            # Extract results theta and time
            plot_rotation = extract(result_rotation, 1)
            plot_time_rotation = extract(result_rotation, 0)
            # Extract results velocity and time
            plot_velocity = extract(result_velocity, 1)
            plot_time_velocity = extract(result_velocity, 0)
            # Extract results landing y and time
            plot_land_y = extract(result_land_y, 1)
            plot_time_land_y = extract(result_land_y, 0)
            # Extract results landing x and time
            plot_land_x = extract(result_land_x, 1)
            plot_time_land_x = extract(result_land_x, 0)

            # Find minimum value reached by each STL formula
            minimum_values = [plot_general_x[0], plot_rotation[0], plot_velocity[0], plot_land_y[0], plot_land_x[0]]

            # Normalized the minimum value respect to the contraint
            minimum_values_normalized = [minimum_values[i] / self.normalization_factors[i] for i in range(5)]

            # Choose the minimum value from the normalized list
            current_robustness = min(minimum_values_normalized)

            # Print the minimum values
            print("")
            print("Minimum values for the tentative n° " + str(ii + 1))
            print(f"minimum_values: {minimum_values}")
            print(f"minimum_values_normalized: {minimum_values_normalized}")
            print(f"Robustness: {current_robustness}")

            # Compute the total robustness as a sum of the minimum values
            robustness.append(current_robustness)

            # Check if falsification is done or not
            done = current_robustness < 0

            # Display the results
            if done:
                # Adding a falsification
                self.counts_falsification += 1

                print("")
                print("Minimum value: " + str(minimum_values))
                print("")

                # Print the initial condition that falsify the trajectory
                print("Initial condition that falsify the trajectory: ")
                print(initial_states[ii])

                print("")
                print("The model was falsified at the " + str(self.falsification_iterations) + "° iterations")

                save_frames_as_gif(frames[ii], f"{self.exp_name}-{self.counts_falsification}")

                # Save initial condition
                self.falsification_list.append([self.counts_falsification] + initial_states[ii][:5])

                # Figure for representing the robustness of the constraints
                plt.figure()
                plt.plot(plot_time_general_x, plot_general_x, label='Robustness X', drawstyle="steps-post")
                plt.plot(plot_time_rotation, plot_rotation, label='Robustness Theta', drawstyle="steps-post")
                plt.plot(plot_time_velocity, plot_velocity, label='Robustness Velocity', drawstyle="steps-post")
                plt.plot(plot_time_land_y, plot_land_y, label='Robustness Landing Y', drawstyle="steps-post")
                plt.plot(plot_time_land_x, plot_land_x, label='Robustness Landing X', drawstyle="steps-post")
                plt.legend()
                plt.show()

        # Print the robustness of all simultions
        print("")
        print("The robustness is: ")
        print(robustness)

        return robustness

    # Function that given a state returns the action with the highest q value
    @tf.function
    def choose_action(self, state):
        q_values = self.model(state, training=False)
        return tf.argmax(q_values[0])

    # Function that simulates the trajectories for a set of initial states
    def simulation(self, env, initial_state):

        # Reset the environment
        env.reset()

        # Initialize the environment in our initial conditions
        env.x0 = initial_state
        state = env.x0
        # Convert the memory to have the proper dimensions
        memory_states = np.expand_dims(state, axis=0)

        frames = []
        # Loop over the steps
        for ii in range(self.simulation_steps):

            # Save frames in the list
            frames.append(env.render())

            # Convert the states to have the proper dimensions
            state = np.expand_dims(state, axis=0).astype(np.float32)

            # Get an action from the model
            action = self.choose_action(state).numpy()
            # Perform the selected action
            next_state, _, done, _, info = env.step(action)

            # Save the state of new step
            memory_states = np.concatenate((memory_states, np.expand_dims(next_state, axis=0)), axis=0)

            # Update the new state
            state = next_state

            # Check for the simulation
            if done:
                break

        return memory_states, frames

    # Function that evaluate the robustness of the initiale states
    # and search for the best individuals
    def evaluation_generation(self, initial_states):

        print("")
        print(str(self.falsification_iterations) + "° iterations")

        # Simulate the new population trajectories
        env = gym.make('LunarLander-v2', render_mode="rgb_array")
        trajectories_and_frames = [self.simulation(env, initial_state=individual) for
                                   individual in initial_states]

        # Close the environment
        env.close()

        # Extract trajectory from the simulation
        trajectories = extract(trajectories_and_frames, 0)

        # Extract frames from the simulation
        frames = extract(trajectories_and_frames, 1)

        # Compute robustness and check for falsification
        robustness = self.get_robustness(trajectories=trajectories,
                                         initial_states=initial_states, frames=frames)

        # Search the index minimum robustness
        _, minimum_robustness = get_best_n_individual(robustness, self.n_offspring_per_parent)

        # Search for the best initial conditions
        self.best_individual.append(
            [
                self.falsification_iterations,
                initial_states[robustness.index(minimum_robustness)][0],
                initial_states[robustness.index(minimum_robustness)][1],
                initial_states[robustness.index(minimum_robustness)][2],
                initial_states[robustness.index(minimum_robustness)][3],
                initial_states[robustness.index(minimum_robustness)][4]
            ]
        )

        return robustness

    # Function that performs the falsification by using the genetic algorithm
    def falsification(self, exp_name, max_falsification_evaluations=1000, n_parents=10, n_offspring_per_parent=3,
                      simulation_steps=1000, mutation_rate=0.2, max_offspring_retries=100):

        # Define the name for saved the data
        self.exp_name = exp_name

        # Inizialize all the value for the genetic algorithm
        self.n_parents = n_parents
        self.n_offspring_per_parent = n_offspring_per_parent
        self.n_pop = n_parents * n_offspring_per_parent
        self.mutation_rate = mutation_rate
        self.max_offspring_retries = max_offspring_retries
        self.simulation_steps = simulation_steps

        # Variable for counting the iterations
        self.counts_falsification = 0
        self.falsification_iterations = 1

        # Define the list to collect the initiale states and best individuals
        self.falsification_list = list()
        self.best_individual = list()

        # Max evaluetion
        max_falsification_iterations = math.ceil(max_falsification_evaluations / self.n_pop)

        # Initialize the first generation of initial states
        initial_states = [self.choose_initial_condition() for _ in range(self.n_pop)]

        # Compute the robustness
        robustness = self.evaluation_generation(initial_states)

        # Loop for falsification
        while self.falsification_iterations <= max_falsification_iterations:
            # Increment the iterations
            self.falsification_iterations += 1

            # Generate a new population
            initial_states = self.genetic_algorith(robustness, initial_states)

            # Compute the robustness
            robustness = self.evaluation_generation(initial_states)

        columns_names = ["x", "y", "v_x", "v_y", "theta"]
        return (pd.DataFrame(self.best_individual, columns=["Count"] + columns_names),
                pd.DataFrame(self.falsification_list, columns=["generation"] + columns_names))

    # Function that save in a gif 5 simulations of the model
    def render_n_simulations(self, n_simulations=5, steps=500, exp_name="NoName"):

        # Initialized the initial conditions
        initial_states = [self.choose_initial_condition() for _ in range(n_simulations)]

        env = gym.make('LunarLander-v2', render_mode="rgb_array")

        self.simulation_steps = steps

        # Run 5 simulations with 5 different initial states
        for ii, initial_state in enumerate(initial_states):
            _, frames_ii = self.simulation(env=env, initial_state=initial_state)
            save_frames_as_gif(frames_ii, f"Simulation_{exp_name}-{ii}")
