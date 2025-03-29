# Update of Max Lapan's Deep RL Hands-on

## DQN Pong

Agent is not learning in the first 200k iterations with the default setup.

Things to try:
    Increase eps_decay_last_frame: From 150,000 to 500,000 or 1,000,000.
    Increase replay_size: From 10,000 to 100,000 or 1,000,000.
    Decrease Target Network Update Frequency: Change sync_target_frames from 1,000 to 5,000 or 10,000.
    Increase Learning Rate: From 1e-4 to 5e-4 or 1e-3.
    Increase Batch Size: From 32 to 64 or 128.
    Clip the reward: Clip the reward to be between -1 and 1.
    Increase eps_final: Try increasing the eps_final to 0.1.
