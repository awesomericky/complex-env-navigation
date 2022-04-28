import pdb

try:
    from ompl import base as ob
    from ompl import geometric as og
except ImportError:
    # if the ompl module is not in the PYTHONPATH assume it is installed in a
    # subdirectory of the parent directory called "py-bindings."
    from os.path import abspath, dirname, join
    import sys
    OMPL_PYBIND_PATH = '' # TODO: PATH_SETUP_REQUIRED # ex) /home/me/Downloads/ompl-1.5.2/py-bindings
    if OMPL_PYBIND_PATH == '':
        raise ValueError("OMPL python binding path is not configured yet.")
    sys.path.insert(0, OMPL_PYBIND_PATH)
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import util as ou

import numpy as np

ALREADY_SET_SEED = False

class Analytic_planner:
    """
    Generate instacle of corresponding class when new environment is generated
    """
    def __init__(self, env, map_size, max_planning_time, min_n_states=40, seed=100):
        self.env = env
        self.map_size = map_size
        self.max_planning_time = max_planning_time  # seconds
        self.min_n_states = min_n_states
        self.path_coordinates = None

        # disable ompl log except error
        ou.setLogLevel(ou.LOG_ERROR)  # LOG_ERROR / LOG_WARN / LOG_INFO / LOG_DEBUG

        global ALREADY_SET_SEED
        if not ALREADY_SET_SEED:
            ou.RNG.setSeed(seed)  # Set seed
            ALREADY_SET_SEED = True

        space = ob.RealVectorStateSpace(2)
        bounds = ob.RealVectorBounds(2)
        try:
            """
            map bound given in matrix form
            """
            self.map_size = self.map_size.astype(np.double)
            bounds.setLow(0, self.map_size[0, 0])
            bounds.setHigh(0, self.map_size[0, 1])
            bounds.setLow(1, self.map_size[1, 0])
            bounds.setHigh(1, self.map_size[1, 1])
        except:
            """
            map size is given in integer
            """
            bounds.setLow(- self.map_size / 2)
            bounds.setHigh(self.map_size / 2)
        space.setBounds(bounds)

        self.si = ob.SpaceInformation(space)
        self.si.setStateValidityChecker(ob.StateValidityCheckerFn(self.isStateValid))
        resolution_size = 0.02  # [m]
        self.si.setStateValidityCheckingResolution(resolution_size / self.si.getMaximumExtent())

        self.start = ob.State(space)
        self.goal = ob.State(space)

        self.planner = og.BITstar(self.si)  # create a planner for the defined space
        self.pdef = ob.ProblemDefinition(self.si)  # create a problem instance

    def isStateValid(self, state):
        """
        Set sphere and check collision

        :return:
        """
        if self.env is None:
            return True
        else:
            safe = not self.env.analytic_planner_collision_check(state[0], state[1])
            return safe
            # return not self.env.analytic_planner_collision_check(state[0], state[1])

    def plan(self, start, goal):
        """

        :param start: (2,) (numpy)
        :param goal: (2,) (numpy)
        :return:
        """
        self.start[0] = start[0]
        self.start[1] = start[1]
        self.goal[0] = goal[0]
        self.goal[1] = goal[1]

        self.pdef = ob.ProblemDefinition(self.si)
        self.pdef.setStartAndGoalStates(self.start, self.goal)
        self.planner.clear()
        self.planner.setProblemDefinition(self.pdef)  # set the problem we are trying to solve for the planner
        self.planner.setup()  # perform setup steps for the planner

        solved = self.planner.solve(self.max_planning_time)

        if solved:
            path = self.pdef.getSolutionPath()
            path.interpolate(self.min_n_states)
            n_states = path.getStateCount()
            path_states = path.getStates()

            self.path_coordinates = np.zeros((n_states, 2))
            for i in range(n_states):
                self.path_coordinates[i][0] = path_states[i][0]
                self.path_coordinates[i][1] = path_states[i][1]
            self.path_coordinates = self.path_coordinates.astype(np.float32)
            return self.path_coordinates.copy()
        else:
            # print("No solution found")
            return None

    def visualize_path(self):
        self.env.visualize_analytic_planner_path(self.path_coordinates)

if __name__ == "__main__":
    map_size = np.array([[-2., 30.], [-1., 31.]])
    planner = Analytic_planner(env=None, map_size=map_size, max_planning_time=10.)
    start = [0., 0.]
    goal = [18., 18.]
    path = planner.plan(start, goal)
    print(path)
