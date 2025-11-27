from ale_py import ALEInterface # import needed for Atari environments for some reason?
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

def rollout(env, p_model, num_steps=10000, seed=0, greedy=False):

    obs, info = env.reset(seed=seed)
    obs_buf, act_buf, lp_buf, rew_buf, done_buf = [], [], [], [], []

    for _ in range(num_steps):
        
        # (num_envs, h, w, c) -> (16, 210, 160, 3)
        obs = obs.astype(np.float32) / 255.0

        obs_tf = tf.convert_to_tensor(obs, dtype=tf.float32)
        logits = p_model(obs_tf, training=False) # (num_envs, num_actions) -> (16, 6)
        
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
        rew_buf.append(reward)
        done_buf.append(dones)

        obs = nxt

    return (np.array(obs_buf, np.float32), # (steps, num_envs, h, w, c) -> (500, 16, 210, 160, 3)
            np.array(act_buf, np.int32), # (steps, num_envs) -> (500, 16)
            np.array(lp_buf, np.float32), # (steps, num_envs) -> (500, 16)
            np.array(rew_buf, np.float32), # (steps, num_envs) -> (500, 16)
            np.array(done_buf, np.float32)) # (steps, num_envs) -> (500, 16)

@tf.function(jit_compile=True)
def train_step(p_model, batch):
    # obs -> (batch_size, h, w, c) -> (512, 210, 160, 3)
    # act -> (batch_size,) -> (512,)
    # old_lps -> (batch_size,) -> (512,)
    # adv -> (batch_size,) -> (512,)
    obs, act, old_lps, adv = batch

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

    return ppo_loss, entropy, policy_loss, approx_kl, clipped_frac, avg_probs

def train_on_dset(p_model, dset, num_epochs=10):
    for _ in range(num_epochs):
        for batch in dset:
            step_out = train_step(p_model, batch)

    return step_out

def train(p_model, num_gens, num_steps, checkpoint_manager):

    for gen in range(num_gens):
        # obs -> (steps, num_envs, h, w, c) -> (500, 16, 210, 160, 3)
        # act -> (steps, num_envs) -> (500, 16)
        # lps -> (steps, num_envs) -> (500, 16)
        # rew -> (steps, num_envs) -> (500, 16)
        # dones -> (steps, num_envs) -> (500, 16)
        obs, act, lps, rew, dones = rollout(env, p_model, num_steps=num_steps, seed=gen)

        obs = tf.reshape(obs, (-1, 210, 160, 3)) # (steps * num_envs, h, w, c) -> (8000, 210, 160, 3)
        act = tf.reshape(act, (-1,)) # (steps * num_envs,) -> (8000,)
        lps = tf.reshape(lps, (-1,)) # (steps * num_envs,) -> (8000,)

        total_reward = tf.reduce_sum(rew, axis=0) # (num_envs,) -> (16,)

        # Mean of total rewards is group baseline
        # Subtract baseline and normalize to get group relative advantage
        # Repeat to use advantage for each step in the rollout
        adv = (total_reward - tf.reduce_mean(total_reward)) / (tf.math.reduce_std(total_reward) + 1e-8) # (num_envs,) -> (16,)
        adv = tf.repeat(adv[None, ...], repeats=num_steps, axis=0) # (steps, num_envs) -> (500, 16)
        adv = tf.reshape(adv, (-1,)) # (steps * num_envs,) -> (8000,)

        avg_reward = tf.reduce_mean(total_reward) # scalar

        dset = (tf.data.Dataset.from_tensor_slices((obs, act, lps, adv))
                .shuffle(buffer_size=num_steps * num_envs)
                .batch(512,
                       drop_remainder=True,
                       deterministic=False,
                       num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE))
        
        (ppo_loss, entropy_bonus, policy_loss,
         approx_kl, clipped_frac, avg_probs) = train_on_dset(p_model, dset, num_epochs=10)

        checkpoint_manager.save()

        print(f'Gen {gen:3d} | Reward: {avg_reward:2.2f} | PPO Loss: {ppo_loss:2.3f} | ' +
              f'Entropy Bonus: {entropy_bonus:2.3f} | Policy Loss: {policy_loss:2.3f} | ' +
              f'Approx KL: {approx_kl:2.3f} | Clipped Fraction: {clipped_frac:2.3f} | ' +
              f'Avg Probs: {avg_probs:2.3f}', flush=True)

p_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(8, 8, 4, activation='relu'),
    tf.keras.layers.Conv2D(16, 4, 2, activation='relu'),
    tf.keras.layers.Conv2D(32, 3, 2, activation='relu'),
    tf.keras.layers.Conv2D(64, 3, 2, activation='relu'),
    tf.keras.layers.Conv2D(128, 3, 2, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(6)
])

p_optimizer = keras.optimizers.Adam(learning_rate=1e-4)
p_model.compile(optimizer=p_optimizer)

p_checkpoint = tf.train.Checkpoint(model=p_model, optimizer=p_optimizer)
p_checkpoint_manager = tf.train.CheckpointManager(p_checkpoint, './grpo_checkpoints', max_to_keep=3)
# p_checkpoint.restore(p_checkpoint_manager.latest_checkpoint)

train(p_model, num_gens=150, num_steps=num_steps, checkpoint_manager=p_checkpoint_manager)

obs, act, lps, rew, dones = rollout(env, p_model, num_steps=num_steps, seed=123, greedy=True)
to_render = (obs[:, 0] * 255.0).astype(np.uint8) # First environment only
imageio.mimsave('grpo_atari.gif', to_render, fps=30, loop=1)