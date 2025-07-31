from leap_c.examples.cylinder.env import CylinderEnv

if __name__ == "__main__":
    print("Instantiate CylinderFlowSolver.")
    env = CylinderEnv(
        render_mode="human", render_method="project", Re=100, Tf=0.05, save_every=0
    )

    print("Reset CylinderFlowSolver.")
    obs, info = env.reset(seed=44)

    terminated = False
    truncated = False
    total_reward = 0
    env.render()

    for i in range(100):  # Increase steps for longer visualization
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if not i % 5:
            env.render()
        # print(f"Current action: {action}")

        if terminated or truncated:
            print(f"Episode finished after {i + 1} timesteps.")
            print(f"Termination: {terminated}, Truncation: {truncated}")
            print(f"Final state (pos): {obs}")
            print(f"Total reward: {total_reward:.2f}")
            if env.render_mode == "human":
                env.render()
            break  # Stop after one episode for this example

    # # Close the environment rendering window
    env.close()
    print("Environment closed.")
