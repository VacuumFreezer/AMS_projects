from ale_py import ALEInterface # needed for Atari environments for some reason :(
import gymnasium as gym
import tensorflow as tf
import keras
from tensorflow_probability import distributions
import numpy as np
import imageio.v2 as imageio

num_envs = 16
num_steps = 500
env = gym.make_vec('ALE/Galaxian-v5', frameskip=4, num_envs=num_envs, vectorization_mode='sync')

timestep = env.reset(seed=0)

def rollout(env, p_model, v_model, num_steps=10000, seed=0, greedy=False):
            
    obs, info = env.reset(seed=seed)
    obs_buf, act_buf, lp_buf, val_buf, rew_buf, done_buf = [], [], [], [], [], []

    for _ in range(num_steps):
        
        # (num_envs, h, w, c) -> (16, 210, 160, 3)
        obs = obs.astype(np.float32) / 255.0

        obs_tf = tf.convert_to_tensor(obs, dtype=tf.float32)
        logits = p_model(obs_tf, training=False) # (num_envs, num_actions) -> (16, 6)
        values = tf.squeeze(v_model(obs_tf, training=False), axis=-1) # (num_envs,) -> (16,)

        dist = distributions.Categorical(logits=logits)

        if greedy:
            actions = tf.argmax(logits, axis=-1) # (num_envs,) -> (16,)
        else:
            actions = dist.sample() # (num_envs,) -> (16,)
        
        chosen_lps = dist.log_prob(actions) # (num_envs,) -> (16,)

        # nxt -> obs shape -> (num_envs, h, w, c) -> (16, 210, 160, 3)
        # reward, terminated, truncated -> (num_envs,) -> (16,)
        nxt, reward, terminated, truncated, _ = env.step(actions.numpy())

        dones = np.logical_or(terminated, truncated).astype(np.float32) # (num_envs,) -> (16,)

        obs_buf.append(obs)
        act_buf.append(actions.numpy())
        lp_buf.append(chosen_lps.numpy())
        val_buf.append(values.numpy())
        rew_buf.append(reward)
        done_buf.append(dones)

        obs = nxt

    return (np.array(obs_buf, np.float32), # (steps, num_envs, h, w, c) -> (500, 16, 210, 160, 3)
            np.array(act_buf, np.int32), # (steps, num_envs) -> (500, 16)
            np.array(lp_buf, np.float32), # (steps, num_envs) -> (500, 16)
            np.array(val_buf, np.float32), # (steps, num_envs) -> (500, 16)
            np.array(rew_buf, np.float32), # (steps, num_envs) -> (500, 16)
            np.array(done_buf, np.float32)) # (steps, num_envs) -> (500, 16)

@tf.function(jit_compile=True)
def compute_gae_and_returns(values, rewards, dones, gamma, lam):

    # (num_steps, num_envs) -> (500, 16)
    advantages = tf.TensorArray(dtype=tf.float32, size=num_steps,
                                element_shape=(num_envs,))

    last_adv = tf.zeros(advantages.element_shape, dtype=tf.float32)
    last_val = values[-1]

    for t in tf.range(num_steps - 1, -1, -1):
        mask = 1.0 - dones[t] # (num_envs,) -> (16,)
        delta = rewards[t] + gamma * last_val * mask - values[t] # (num_envs,) -> (16,)
        last_adv = delta + gamma * lam * last_adv * mask # (num_envs,) -> (16,)
        advantages = advantages.write(t, last_adv)
        last_val = values[t]

    advantages = advantages.stack()

    returns = advantages + values # (num_steps, num_envs) -> (500, 16)

    advantages = (
        (advantages - tf.reduce_mean(advantages)) /
        (tf.math.reduce_std(advantages) + 1e-8)
    )

    return advantages, returns

@tf.function(jit_compile=True)
def train_step(p_model, v_model, batch):
    # obs -> (batch_size, h, w, c) -> (512, 210, 160, 3)
    # act -> (batch_size,) -> (512,)
    # old_lps -> (batch_size,) -> (512,)
    # adv -> (batch_size,) -> (512,)
    obs, act, old_lps, adv, ret = batch

    with tf.GradientTape() as p_tape:
        logits = p_model(obs, training=True) # (batch_size, num_actions) -> (512, 6)
        dist = distributions.Categorical(logits=logits)
        new_lps = dist.log_prob(act) # (batch_size,) -> (512,)
        entropy = tf.reduce_mean(dist.entropy()) # scalar

        ratio = tf.exp(new_lps - old_lps) # (batch_size,) -> (512,)
        clipped_ratio = tf.clip_by_value(ratio, 0.8, 1.2) # (batch_size,) -> (512,)
        ppo_loss = -tf.reduce_mean(tf.minimum(ratio * adv, clipped_ratio * adv)) # scalar

        policy_loss = ppo_loss - 0.01 * entropy # scalar

    p_grads = p_tape.gradient(policy_loss, p_model.trainable_variables)
    p_model.optimizer.apply_gradients(zip(p_grads, p_model.trainable_variables))

    approx_kl = tf.reduce_mean(old_lps - new_lps) # scalar
    clipped_frac = tf.reduce_mean(tf.cast(ratio != clipped_ratio, tf.float32)) # scalar
    avg_probs = tf.reduce_mean(tf.exp(new_lps)) # scalar

    with tf.GradientTape() as v_tape:
        values = tf.squeeze(v_model(obs, training=True), axis=-1) # (batch_size,) -> (512,)
        value_loss = tf.reduce_mean((ret - values) ** 2) # scalar

    v_grads = v_tape.gradient(value_loss, v_model.trainable_variables)
    v_model.optimizer.apply_gradients(zip(v_grads, v_model.trainable_variables))

    return ppo_loss, entropy, policy_loss, value_loss, approx_kl, clipped_frac, avg_probs

def train_on_dset(p_model, v_model, dset, num_epochs=10):
    for _ in range(num_epochs):
        for batch in dset:
            step_out = train_step(p_model, v_model, batch)

    return step_out

def train(p_model, v_model, num_gens, num_steps=500):

    for gen in range(num_gens):
        # obs -> (steps, num_envs, h, w, c) -> (500, 16, 210, 160, 3)
        # act -> (steps, num_envs) -> (500, 16)
        # lps -> (steps, num_envs) -> (500, 16)
        # rew -> (steps, num_envs) -> (500, 16)
        # vals -> (steps, num_envs) -> (500, 16)
        # dones -> (steps, num_envs) -> (500, 16)
        obs, act, lps, vals, rew, dones = rollout(env, p_model, v_model, num_steps=num_steps, seed=gen)
        # adv -> (steps, num_envs) -> (500, 16)
        # ret -> (steps, num_envs) -> (500, 16)
        adv, ret = compute_gae_and_returns(vals, rew, dones, gamma=0.99, lam=0.95)

        obs = tf.reshape(obs, (-1, 210, 160, 3)) # (steps * num_envs, h, w, c) -> (8000, 210, 160, 3)
        act = tf.reshape(act, (-1,)) # (steps * num_envs,) -> (8000,)
        lps = tf.reshape(lps, (-1,)) # (steps * num_envs,) -> (8000,)
        adv = tf.reshape(adv, (-1,)) # (steps * num_envs,) -> (8000,)
        ret = tf.reshape(ret, (-1,)) # (steps * num_envs,) -> (8000,)

        total_reward = tf.reduce_sum(rew, axis=0) # (num_envs,) -> (16,)
        avg_reward = tf.reduce_mean(total_reward) # scalar

        dset = (tf.data.Dataset.from_tensor_slices((obs, act, lps, adv, ret))
                .shuffle(buffer_size=num_steps * num_envs)
                .batch(512,
                       drop_remainder=True,
                       deterministic=False,
                       num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE))
        
        (ppo_loss, entropy_bonus, policy_loss, value_loss,
         approx_kl, clipped_frac, avg_probs) = train_on_dset(p_model, v_model, dset, num_epochs=10)

        print(f"Gen {gen:3d} | Reward: {avg_reward:.2f} | PPO Loss: {ppo_loss:.3f} | " +
              f"Entropy Bonus: {entropy_bonus:.3f} | Policy Loss: {policy_loss:.3f} | " +
              f"Value Loss: {value_loss:.3f} | Approx KL: {approx_kl:.3f} | " +
              f"Clipped Fraction: {clipped_frac:.3f} | Avg Probs: {avg_probs:.3f}", flush=True)

p_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(8, 8, 4, activation="relu"),
    tf.keras.layers.Conv2D(16, 4, 2, activation="relu"),
    tf.keras.layers.Conv2D(32, 3, 2, activation="relu"),
    tf.keras.layers.Conv2D(64, 3, 2, activation="relu"),
    tf.keras.layers.Conv2D(128, 3, 2, activation="relu"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(6)
])

p_optimizer = keras.optimizers.Adam(learning_rate=1e-4)
p_model.compile(optimizer=p_optimizer)

v_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(8, 8, 4, activation="relu"),
    tf.keras.layers.Conv2D(16, 4, 2, activation="relu"),
    tf.keras.layers.Conv2D(32, 3, 2, activation="relu"),
    tf.keras.layers.Conv2D(64, 3, 2, activation="relu"),
    tf.keras.layers.Conv2D(128, 3, 2, activation="relu"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(1)
])

v_optimizer = keras.optimizers.Adam(learning_rate=1e-4)
v_model.compile(optimizer=v_optimizer)

p_checkpoint = tf.train.Checkpoint(model=p_model, optimizer=p_optimizer)
p_checkpoint_manager = tf.train.CheckpointManager(p_checkpoint, './ppo_checkpoints', max_to_keep=3)
# p_checkpoint.restore(p_checkpoint_manager.latest_checkpoint)

v_checkpoint = tf.train.Checkpoint(model=v_model, optimizer=v_optimizer)
v_checkpoint_manager = tf.train.CheckpointManager(v_checkpoint, './ppo_checkpoints', max_to_keep=3)
# v_checkpoint.restore(v_checkpoint_manager.latest_checkpoint)

train(p_model, v_model, num_gens=50, num_steps=num_steps)

obs, act, lps, vals, rew, dones = rollout(env, p_model, v_model, num_steps=num_steps, seed=123, greedy=True)
to_render = (obs[:, 0] * 255.0).astype(np.uint8) # First environment only
imageio.mimsave('ppo_atari.gif', to_render, fps=30, loop=1)