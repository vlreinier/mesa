from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from mesa.batchrunner import BatchRunner

import matplotlib.pyplot as plt
import numpy as np


def compute_gini(model):
    agent_wealths = [agent.wealth for agent in model.schedule.agents]
    x = sorted(agent_wealths)
    N = model.num_agents
    B = sum( xi * (N-i) for i, xi in enumerate(x) ) / (N*sum(x))
    return 1 + (1/N) - 2*B


class MoneyModel(Model):
    """A model with some number of agents."""
    def __init__(self, N, width, height):
        self.num_agents = N
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.running = True

        # Create agents
        for i in range(self.num_agents):
            a = MoneyAgent(i, self)
            self.schedule.add(a)
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

        # data collector
        self.datacollector = DataCollector(
            model_reporters={"Gini": compute_gini},  # `compute_gini` defined above
            agent_reporters={"Wealth": "wealth"})

    def step(self):
        '''Advance the model by one step.'''
        self.datacollector.collect(self)
        self.schedule.step()


class MoneyAgent(Agent):
    """ An agent with fixed initial wealth."""
    def __init__(self, unique_id, model):
        super(MoneyAgent, self).__init__(unique_id, model)
        self.wealth = 1

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False)
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def give_money(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])

        # own input for changing code a little bit
        # give away to poorest neighbour instead of random one
        if len(cellmates) > 1:
            cellmates_and_wealths = {mate.wealth: mate for mate in cellmates}
            poorest_cellmate = cellmates_and_wealths[min(cellmates_and_wealths)]
            poorest_cellmate.wealth += 1
            self.wealth -= 1

        # if len(cellmates) > 1:
        #     other = self.random.choice(cellmates)
        #     other.wealth += 1
        #     self.wealth -= 1

    def step(self):
        self.move()
        if self.wealth > 0:
            self.give_money()


model = MoneyModel(50, 10, 10)
for i in range(100):
    model.step()

agent_counts = np.zeros((model.grid.width, model.grid.height))
for cell in model.grid.coord_iter():
    cell_content, x, y = cell
    agent_count = len(cell_content)
    agent_counts[x][y] = agent_count
plt.imshow(agent_counts, interpolation='nearest')
plt.colorbar()
plt.show()


# Data collector
gini = model.datacollector.get_model_vars_dataframe()
plt.plot(gini)
plt.show()

# agent_wealth = model.datacollector.get_agent_vars_dataframe()
# print(agent_wealth)


# batch runner
# one_agent_wealth = agent_wealth.xs(14, level="AgentID")
# plt.plot(one_agent_wealth.Wealth)
# #plt.show()
#
#
# fixed_params = {
#     "width": 10,
#     "height": 10
# }
# variable_params = {"N": range(10, 500, 10)}
#
# # The variables parameters will be invoke along with the fixed parameters allowing for either or both to be honored.
# batch_run = BatchRunner(
#     MoneyModel,
#     variable_params,
#     fixed_params,
#     iterations=5,
#     max_steps=100,
#     model_reporters={"Gini": compute_gini}
# )
#
# batch_run.run_all()
#
# run_data = batch_run.get_model_vars_dataframe()
# plt.scatter(run_data.N, run_data.Gini)
# plt.show()