from enum import Enum
import numpy as np


class State():

    def __init__(self, row=-1, column=-1):
        self.row = row
        self.column = column

    def __repr__(self):
        return f"<State: [{self.row}, {self.column}]>"

    def clone(self):
        """Return the same state."""
        return State(self.row, self.column)

    def __hash__(self):
        return hash((self.row, self.column))

    def __eq__(self, other):
        return self.row == other.row and self.column == other.column


class Action(Enum):
    UP = 1
    DOWN = -1
    LEFT = 2
    RIGHT = -2


class Environment():

    def __init__(self, grid, move_prob=0.8):
        # grid is 2d-array. Its values are treated as an attribute.
        # Kinds of attribute is following.
        #  0: ordinary cell
        #  -1: damage cell (game end)
        #  1: reward cell (game end)
        #  9: block cell (can't locate agent)
        self.grid = grid
        self.agent_state = State()

        # Default reward is minus. Just like a poison swamp.
        # It means the agent has to reach the goal fast!
        self.default_reward = -0.04

        # Agent can move to a selected direction in move_prob.
        # It means the agent will move different direction
        # in (1 - move_prob).
        self.move_prob = move_prob
        self.reset()

    @property
    def row_length(self):
        """Return the number of rows."""
        return len(self.grid)

    @property
    def column_length(self):
        """Return the number of columns."""
        return len(self.grid[0])

    @property
    def actions(self):
        """Return all possible actions."""
        return [Action.UP, Action.DOWN,
                Action.LEFT, Action.RIGHT]

    @property
    def states(self):
        states = []
        for row in range(self.row_length):
            for column in range(self.column_length):
                # Block cells are not included to the state.
                if self.grid[row][column] != 9:
                    states.append(State(row, column))
        return states

    def transit_func(self, state: State, action):
        """ 
        次のstateの候補と，そのstateになる確率をもつdictを返す.
        ここから移動できない時は空のdictを返す．
        """
        transition_probs = {}
        if not self.can_action_at(state):
            # Already on the terminal cell.
            return transition_probs

        opposite_direction = Action(action.value * -1)

        for a in self.actions:
            # aが選ばれるprobを求める
            prob = 0
            if a == action:
                prob = self.move_prob
            elif a != opposite_direction:
                prob = (1 - self.move_prob) / 2

            # aで行動した場合の次のstateを求める
            next_state = self._move(state, a)
            
            # そのstateになる確率としてprobを加える
            if next_state not in transition_probs:
                transition_probs[next_state] = prob
            else:
                transition_probs[next_state] += prob

        return transition_probs

    def can_action_at(self, state):
        """Check if agent can move at the state."""
        if self.grid[state.row][state.column] == 0:
            return True
        else:
            return False

    def _move(self, state: State, action):
        if not self.can_action_at(state):
            raise Exception("Can't move from here!")

        next_state = state.clone()

        # Execute an action (move).
        if action == Action.UP:
            next_state.row -= 1
        elif action == Action.DOWN:
            next_state.row += 1
        elif action == Action.LEFT:
            next_state.column -= 1
        elif action == Action.RIGHT:
            next_state.column += 1

        # Check whether a state is out of the grid.
        if not (0 <= next_state.row < self.row_length):
            next_state = state
        if not (0 <= next_state.column < self.column_length):
            next_state = state

        # Check whether the agent bumped a block cell.
        if self.grid[next_state.row][next_state.column] == 9:
            next_state = state

        return next_state

    def reward_func(self, state):
        """ そのstateでのrewardと，ゲームが終了したかどうかを返す """
        reward = self.default_reward
        done = False

        # Check an attribute of next state.
        attribute = self.grid[state.row][state.column]
        if attribute == 1:
            # Get reward! and the game ends.
            reward = 1
            done = True
        elif attribute == -1:
            # Get damage! and the game ends.
            reward = -1
            done = True

        return reward, done

    def reset(self):
        """ Environmentが持つagentの位置を初期化する """
        
        # Locate the agent at lower left corner.
        self.agent_state = State(self.row_length - 1, 0)
        return self.agent_state

    def step(self, action):
        """
        environmentを1step進める
        
        Returns:
            tuple: 以下の値をタプルで返します。
            - next_state (State): 次のstate
            - reward_of_next_state (float): next_stateで得られるreward
            - done (bool): ゲームが終了した
        """
        next_state, reward_of_next_state, done = self.transit(self.agent_state, action)
        if next_state is not None:
            self.agent_state = next_state

        return next_state, reward_of_next_state, done

    def transit(self, state, action):
        """
        今いるStateでActionを選択したときの次のstate，そのstateのreward，doneを返す

        Args:
            state (State): 今いるstate
            action (Action): 選択したaction

        Returns:
            tuple: 以下の値をタプルで返します。
            - next_state (State): 次のstate
            - reward_of_next_state (float): reward
            - done (bool): ゲームが終了した
        """
        transition_probs = self.transit_func(state, action)
        if len(transition_probs) == 0:
            return None, None, True

        next_states = []
        probs = []
        for s in transition_probs:
            next_states.append(s)
            probs.append(transition_probs[s])

        next_state = np.random.choice(next_states, p=probs)
        reward_of_next_state , done = self.reward_func(next_state)
        return next_state, reward_of_next_state, done
