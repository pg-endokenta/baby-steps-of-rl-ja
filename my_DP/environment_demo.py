from agent import Agent
from environment import Environment



def main():
    # Make grid environment.
    grid = [
        [0, 0, 0, 1],
        [0, 9, 0, -1],
        [0, 0, 0, 0]
    ]
    env = Environment(grid)
    agent = Agent(env)

    # Try 10 game.
    for i in range(10):
        # Initialize position of agent.
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.policy(state)
            
            # ここでアクションによる移動先のstateの決定，そのstateのrewardの取得，ゲーム終了かどうかの判定が行われる
            next_state, reward_of_next_state, done = env.step(action)
            
            total_reward += reward_of_next_state
            # agent側でもstateを更新しておく
            state = next_state

        print(f"Episode {i}: Agent gets {total_reward: .2f} reward.")


if __name__ == "__main__":
    main()
