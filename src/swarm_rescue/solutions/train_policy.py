# Initialize Environment

# We use RLEnv class are our environment wrapper and see if if any accessible compoentns of sim can be implemented in gym style
# Env.make()

# episode , lr , parmaeters etc

# initialize policy (implemented elsewhere)
# We use pytorch

# We probably need to learn off policy so need to store all transitions in replay buffer.

# initialize optimizer loss function

# Training Loop
# for episode in range(num_episodes):
    # reset environment
    # for t in range(max_timesteps):
        # select action using policy
        # execute action in environment
        # usualy here comes env.step to get next state reward done and info. Tricky, here since we are one stpe behind.
        # store transition in replay buffer
        # sample random minibatch from replay buffer
        # compute target and current Q values
        # compute loss
        # perform gradient descent step
    # log progress
    # save model periodically

