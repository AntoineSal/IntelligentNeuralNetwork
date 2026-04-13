# @title ⛏️ CRAFTER BENCHMARK: INN (Vision Only) + Recurrent PPO
# @markdown Architecture : 6 Neurones LSTM indépendants + Attention Tous-vers-Tous.
# @markdown Mode : Vision Only (Apprend à lire l'UI directement sur l'image pour robustesse).

import os
import random
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# Auto-install dependencies
try:
    import crafter
except ImportError:
    print("Installing Crafter...")
    os.system("pip install -q crafter gymnasium")
    import crafter

import gymnasium as gym

# ==========================================
# CONFIGURATION
# ==========================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Running on {DEVICE}")

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Hyperparameters (Tuned for Crafter SOTA - Turbo Mode)
NUM_ENVS = 16                # Parallel environments
TOTAL_STEPS = 2_000_000      # Total frames
ROLLOUT_STEPS = 256          # Increased: 16*256 = 4096 steps per update (Better Gradient)
MINIBATCH_SEQ_LEN = 32       # TBPTT Sequence length
NUM_MINIBATCHES = 8          # Increased minibatches for 4096 steps
PPO_EPOCHS = 4               # Optimization epochs
LR = 7e-5                    # Increased for faster policy learning
LR_CRITIC = 5e-5             # Stable critic LR
RND_SCALE = 0.005            # Boosted Curiosity (x5) to force exploration
GAMMA = 0.97                 # Reduced Gamma: Focus on immediate survival/crafting stability
GAE_LAMBDA = 0.92            # Reduced Lambda for less variance
CLIP_EPS = 0.25              # Increased Clipping: Allow bolder updates
ENT_COEF = 0.01              # Entropy bonus
MAX_GRAD_NORM = 0.5
SAVE_DIR = "./inn_crafter_checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# INN Specs
N_NEURONS = 6
D_MODEL = 128                # Dimension per neuron

# ==========================================
# ENVIRONMENT WRAPPER (VISION ONLY)
# ==========================================

class CrafterVisionWrapper(gym.Env):
    """
    Wrapper ultra-robuste qui ne dépend QUE de l'image.
    Évite toutes les erreurs d'attributs internes de Crafter.
    """
    def __init__(self, seed):
        self._env = crafter.Env(seed=seed)
        
        # Image: (64, 64, 3)
        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8),
        })
        self.action_space = gym.spaces.Discrete(17)
        
    def reset(self, seed=None, options=None):
        obs = self._env.reset()
        return {'image': obs}, {}

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        return {'image': obs}, reward, done, False, info

def make_env(idx):
    def _thunk():
        env = CrafterVisionWrapper(seed=SEED + idx)
        return env
    return _thunk

# ==========================================
# ARCHITECTURE: INN (All-to-All)
# ==========================================

class DualVisionEncoder(nn.Module):
    """
    Encodes image into two streams:
    1. Inventory Head: Finer details (icons, text)
    2. Environment Head: Larger receptive field (objects, terrain)
    """
    def __init__(self, d_model):
        super().__init__()
        
        # Inventory Head (High Res, Small Kernels)
        self.inv_head = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), 
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, d_model), nn.ReLU()
        )
        
        # Environment Head (Low Res, Large Kernels)
        self.env_head = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, d_model), nn.ReLU()
        )

    def forward(self, x):
        # x: [B, 3, H, W] already permuted and normalized
        inv_feat = self.inv_head(x)
        env_feat = self.env_head(x)
        return inv_feat, env_feat

class INN_Neuron(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ln_in = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=2, batch_first=True)
        self.lstm = nn.LSTMCell(d_model * 2, d_model) 
        self.ln_out = nn.LayerNorm(d_model)

    def forward(self, h, c, all_states, external_input=None):
        # 1. Communication (Query = h, Key/Value = all_states)
        query = h.unsqueeze(1)
        context, _ = self.attn(query, all_states, all_states) 
        context = context.squeeze(1)

        # 2. Input Fusion
        if external_input is None:
            external_input = torch.zeros_like(h)
        
        rnn_input = torch.cat([context, external_input], dim=-1)
        
        # 3. Recurrence
        new_h, new_c = self.lstm(rnn_input, (h, c))
        return self.ln_out(new_h), new_c

class INN_Brain(nn.Module):
    def __init__(self, num_actions=17):
        super().__init__()
        self.n_neurons = N_NEURONS
        self.d_model = D_MODEL
        
        # Dual Vision Encoder
        self.dual_encoder = DualVisionEncoder(D_MODEL)
        
        # Learnable Gating per Neuron (Initialized to NEUTRAL)
        # Value 0.0 means sigmoid(0.0) = 0.5 (equal attention to env and inv)
        self.gate_logits = nn.Parameter(torch.zeros(self.n_neurons))
        
        # Per-Neuron Projections
        self.inv_projs = nn.ModuleList([nn.Linear(D_MODEL, D_MODEL) for _ in range(self.n_neurons)])
        self.env_projs = nn.ModuleList([nn.Linear(D_MODEL, D_MODEL) for _ in range(self.n_neurons)])
        
        self.neurons = nn.ModuleList([INN_Neuron(D_MODEL) for _ in range(N_NEURONS)])
        self.actor = nn.Linear(D_MODEL, num_actions)
        self.critic = nn.Linear(D_MODEL, 1)
        
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.orthogonal_(param, gain=np.sqrt(2))

    def get_initial_state(self, batch_size):
        return [
            (torch.zeros(batch_size, self.d_model).to(DEVICE),
             torch.zeros(batch_size, self.d_model).to(DEVICE))
            for _ in range(self.n_neurons)
        ]

    def forward(self, obs, state, done=None):
        B = obs['image'].shape[0]
        
        # 1. Perception (Dual Stream)
        img = obs['image'].permute(0, 3, 1, 2).float() / 255.0 - 0.5
        inv_feat, env_feat = self.dual_encoder(img)
        
        # Modality Dropout (Training only - Per Sample)
        if self.training:
            # Create masks [B, 1]
            inv_mask = (torch.rand(inv_feat.size(0), 1, device=inv_feat.device) >= 0.1).float()
            env_mask = (torch.rand(env_feat.size(0), 1, device=env_feat.device) >= 0.1).float()
            
            inv_feat = inv_feat * inv_mask
            env_feat = env_feat * env_mask

        # 2. Context
        prev_h_stacked = torch.stack([s[0] for s in state], dim=1) 
        
        # 3. Reset Masking
        if done is not None:
            mask = 1.0 - done.float().unsqueeze(-1)
            state = [(h * mask, c * mask) for (h, c) in state]
            prev_h_stacked = prev_h_stacked * mask.unsqueeze(1)

        # 4. Update Neurons with Gated Input
        g = torch.sigmoid(self.gate_logits).view(1, self.n_neurons, 1) # [1, N, 1]
        
        new_state = []
        for i in range(self.n_neurons):
            # Compute specific input for this neuron
            # ext = g[i] * inv + (1-g[i]) * env
            proj_inv = self.inv_projs[i](inv_feat)
            proj_env = self.env_projs[i](env_feat)
            
            gate_val = g[:, i, :]
            ext_input = gate_val * proj_inv + (1.0 - gate_val) * proj_env
            
            h, c = state[i]
            new_h, new_c = self.neurons[i](h, c, prev_h_stacked, ext_input)
            new_state.append((new_h, new_c))
            
        # 5. Readout
        final_h_stacked = torch.stack([s[0] for s in new_state], dim=1)
        global_representation = torch.mean(final_h_stacked, dim=1)
        
        logits = self.actor(global_representation)
        value = self.critic(global_representation)
        
        return logits, value, new_state

# ==========================================
# STORAGE BUFFER
# ==========================================

class RecurrentBuffer:
    def __init__(self, num_steps, num_envs, obs_shape, n_neurons, d_model, device):
        self.obs_img = torch.zeros((num_steps, num_envs) + obs_shape, dtype=torch.uint8, device=device)
        self.actions = torch.zeros((num_steps, num_envs), dtype=torch.long).to(device) # Corrected to long
        self.logprobs = torch.zeros((num_steps, num_envs)).to(device)
        self.rewards = torch.zeros((num_steps, num_envs)).to(device)
        self.ext_rewards = torch.zeros((num_steps, num_envs)).to(device)
        self.dones = torch.zeros((num_steps, num_envs)).to(device)
        self.values = torch.zeros((num_steps, num_envs)).to(device)
        
        # Store RNN States: [Steps, Envs, N_Neurons, D_Model]
        self.h_states = torch.zeros((num_steps, num_envs, n_neurons, d_model)).to(device)
        self.c_states = torch.zeros((num_steps, num_envs, n_neurons, d_model)).to(device)

    def store(self, step, obs, action, logprob, reward, ext_reward, done, value, state):
        # Secure storage
        img = torch.from_numpy(obs['image']).to(dtype=torch.uint8, device=DEVICE)
        self.obs_img[step].copy_(img)
        self.actions[step].copy_(action.to(DEVICE)) # No float cast for actions
        self.logprobs[step].copy_(logprob.detach().to(DEVICE))
        self.rewards[step].copy_(reward.detach().to(DEVICE))
        self.ext_rewards[step].copy_(ext_reward.detach().to(DEVICE))
        self.dones[step].copy_(torch.tensor(done, dtype=torch.float32, device=DEVICE))
        self.values[step].copy_(value.detach().to(DEVICE))
        
        # Store Hidden States
        # state is a list of (h, c) tuples for each neuron
        # h shape: [Envs, D_Model]
        for i, (h, c) in enumerate(state):
            self.h_states[step, :, i].copy_(h.detach())
            self.c_states[step, :, i].copy_(c.detach())

# ==========================================
# RND MODULE (Intrinsic Curiosity)
# ==========================================

class RNDModule(nn.Module):
    def __init__(self):
        super().__init__()
        # Target Network (Fixe)
        self.target = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)), # Force output spatial dim to 2x2
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 512)
        )
        # Predictor Network (Trainable)
        self.predictor = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)), # Force output spatial dim to 2x2
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        # Freeze target
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, obs):
        # obs: [B, 64, 64, 3] -> permute inside
        x = obs.permute(0, 3, 1, 2).float() / 255.0 - 0.5
        with torch.no_grad():
            target_feat = self.target(x)
        pred_feat = self.predictor(x)
        return target_feat, pred_feat

# ==========================================
# TRAINING LOOP
# ==========================================

if __name__ == "__main__":
    envs = gym.vector.AsyncVectorEnv([make_env(i) for i in range(NUM_ENVS)])
    
    # Agent & RND
    agent = INN_Brain().to(DEVICE)
    rnd = RNDModule().to(DEVICE)
    
    print(f"🧠 Model Parameters: {sum(p.numel() for p in agent.parameters())}")
    
    # Separate Optimizers (Stability & Gate Boost)
    # Group parameters
    gate_params = [p for n,p in agent.named_parameters() if "gate_logits" in n]
    critic_params = [p for n,p in agent.named_parameters() if "critic" in n]
    actor_params = [p for n,p in agent.named_parameters() if "critic" not in n and "gate_logits" not in n]
    
    # Optimizer with parameter groups
    opt_actor = torch.optim.Adam([
        {'params': actor_params, 'lr': LR},
        {'params': gate_params, 'lr': LR * 10.0} # Boost Gate LR to unfreeze it
    ], eps=1e-5)
    
    opt_critic = torch.optim.Adam(critic_params, lr=LR_CRITIC, eps=1e-5)
    rnd_optimizer = torch.optim.Adam(rnd.predictor.parameters(), lr=LR, eps=1e-5)
    
    buffer = RecurrentBuffer(ROLLOUT_STEPS, NUM_ENVS, (64,64,3), N_NEURONS, D_MODEL, DEVICE)

    global_step = 0
    next_obs, _ = envs.reset()
    next_done = torch.zeros(NUM_ENVS).to(DEVICE)
    next_state = agent.get_initial_state(NUM_ENVS)
    
    # RND Running Stats
    int_reward_mean = 0.0
    int_reward_std = 1.0
    
    # 10M Steps SOTA Run
    TOTAL_STEPS = 10_000_000
    num_updates = TOTAL_STEPS // (NUM_ENVS * ROLLOUT_STEPS)
    print("🏁 STARTING TRAINING (INN Vision-Only Dual-Stream + RND Stabilized)...")

    try:
        for update in range(1, num_updates + 1):
            initial_rollout_state = [(h.detach(), c.detach()) for (h, c) in next_state]
            
            for step in range(ROLLOUT_STEPS):
                global_step += NUM_ENVS
                
                # 1. Save state BEFORE forward (Critical for PPO-RNN alignment)
                current_state_to_store = [(h.detach(), c.detach()) for (h, c) in next_state]
                
                obs_tensor = {'image': torch.tensor(next_obs['image']).to(DEVICE)}
                
                # 2. Agent Action
                with torch.no_grad():
                    logits, value, next_state_out = agent(obs_tensor, next_state, next_done)
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                    logprob = dist.log_prob(action)
                
                # 3. Environment Step
                real_next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
                done = np.logical_or(terminated, truncated)
                
                # 4. Intrinsic Reward (RND) & Online Normalization
                with torch.no_grad():
                    next_img = torch.from_numpy(real_next_obs['image']).to(DEVICE)
                    t_feat, p_feat = rnd(next_img)
                    r_int = (t_feat - p_feat).pow(2).sum(dim=1) / 2.0
                    
                    # Update Running Stats (Online)
                    batch_mean = r_int.mean().item()
                    batch_std = r_int.std().item()
                    int_reward_mean = 0.999 * int_reward_mean + 0.001 * batch_mean
                    int_reward_std = 0.999 * int_reward_std + 0.001 * batch_std
                    
                    # Normalize immediately (Scale by STD only to keep positivity)
                    r_int_norm = r_int / (int_reward_std + 1e-8)
                
                # 5. Combine Rewards
                r_ext = torch.tensor(reward).to(DEVICE)
                
                # Clip intrinsic reward contribution to avoid destabilizing PPO
                # Cap bonus at 0.5 (practical limit)
                r_bonus = torch.clamp(RND_SCALE * r_int_norm, 0.0, 0.5)
                r_total = r_ext + r_bonus
                
                # 6. Store (Aligned: state used for action -> action -> reward resulting)
                buffer.store(step, next_obs, action, logprob, r_total, r_ext, done, value.flatten(), current_state_to_store)
                
                # 7. Update Pointers
                next_obs = real_next_obs
                next_done = torch.tensor(done).to(DEVICE)
                next_state = next_state_out # This becomes the input state for next step

            # Update RND Predictor on FULL SHUFFLED BUFFER (Corrected)
            b_obs_rnd = buffer.obs_img.view(-1, 64, 64, 3)
            idx = torch.randperm(b_obs_rnd.size(0))[:min(4096, b_obs_rnd.size(0))]
            t_feat, p_feat = rnd(b_obs_rnd[idx])
            rnd_loss = (t_feat - p_feat).pow(2).mean()
            
            rnd_optimizer.zero_grad()
            rnd_loss.backward()
            rnd_optimizer.step()

            with torch.no_grad():
                 obs_tensor = {'image': torch.tensor(next_obs['image']).to(DEVICE)}
                 _, next_value, _ = agent(obs_tensor, next_state, next_done)
                 advantages = torch.zeros_like(buffer.rewards).to(DEVICE)
                 lastgaelam = 0
                 for t in reversed(range(ROLLOUT_STEPS)):
                     if t == ROLLOUT_STEPS - 1:
                         nextnonterminal = 1.0 - next_done.float()
                         nextvalues = next_value.flatten()
                     else:
                         nextnonterminal = 1.0 - buffer.dones[t + 1]
                         nextvalues = buffer.values[t + 1]
                     
                     delta = buffer.rewards[t] + GAMMA * nextvalues * nextnonterminal - buffer.values[t]
                     advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
                 
                 returns = advantages + buffer.values

            # Normalize Advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Recurrent PPO Update
            b_obs_img = buffer.obs_img
            b_actions = buffer.actions
            b_logprobs = buffer.logprobs
            b_advantages = advantages
            b_returns = returns
            b_dones = buffer.dones
            b_values = buffer.values # Needed for proper clipping
            
            # Retrieve stored states
            b_h_states = buffer.h_states
            b_c_states = buffer.c_states
            
            total_v_loss = 0
            total_p_loss = 0
            
            for _ in range(PPO_EPOCHS):
                for t in range(0, ROLLOUT_STEPS, MINIBATCH_SEQ_LEN):
                    end_t = min(t + MINIBATCH_SEQ_LEN, ROLLOUT_STEPS)
                    c_obs_img = b_obs_img[t:end_t]
                    c_actions = b_actions[t:end_t]
                    c_logprobs = b_logprobs[t:end_t]
                    c_returns = b_returns[t:end_t]
                    c_advs = b_advantages[t:end_t]
                    c_dones = b_dones[t:end_t]
                    c_old_values = b_values[t:end_t] # Old values for clipping
                    
                    # Reconstruct Initial State for this Chunk from Buffer
                    current_h = b_h_states[t]
                    current_c = b_c_states[t]
                    
                    current_chunk_state = []
                    for n_idx in range(N_NEURONS):
                        current_chunk_state.append((current_h[:, n_idx], current_c[:, n_idx]))
                    
                    new_logprobs, new_values, new_entropies = [], [], []
                    
                    for k in range(end_t - t):
                        step_obs = {'image': c_obs_img[k]}
                        step_done = c_dones[k]
                        logits, val, current_chunk_state = agent(step_obs, current_chunk_state, step_done)
                        dist = Categorical(logits=logits)
                        new_logprobs.append(dist.log_prob(c_actions[k]))
                        new_values.append(val.flatten())
                        new_entropies.append(dist.entropy())
                    
                    new_logprobs = torch.stack(new_logprobs)
                    new_values = torch.stack(new_values)
                    new_entropies = torch.stack(new_entropies)
                    
                    # Loss
                    logratio = new_logprobs - c_logprobs
                    ratio = logratio.exp()
                    pg_loss1 = -c_advs * ratio
                    pg_loss2 = -c_advs * torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    
                    # Value Clipping (Correct Implementation)
                    # new_values and c_old_values must match shape
                    v_loss_unclipped = (new_values - c_returns) ** 2
                    v_clipped = c_old_values + torch.clamp(new_values - c_old_values, -CLIP_EPS, CLIP_EPS)
                    v_loss_clipped = (v_clipped - c_returns) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    
                    entropy_loss = new_entropies.mean()
                    
                    loss_actor = pg_loss - entropy_loss * ENT_COEF
                    loss_critic = v_loss
                    
                    # Combined Loss for Single Backward Pass (More Efficient, No Graph Leak)
                    loss_total = loss_actor + 0.5 * loss_critic

                    opt_actor.zero_grad()
                    opt_critic.zero_grad()
                    
                    loss_total.backward()
                    
                    nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                    
                    opt_actor.step()
                    opt_critic.step()
                    
                    total_v_loss += v_loss.item()

            mean_total_reward = buffer.rewards.sum(dim=0).mean().item()
            mean_game_reward = buffer.ext_rewards.sum(dim=0).mean().item()
            mean_bonus = mean_total_reward - mean_game_reward
            gate_mean = torch.sigmoid(agent.gate_logits).mean().item()
            
            print(f"Step {global_step} | Avg Total: {mean_total_reward:.2f} | Avg Game: {mean_game_reward:.2f} | RND Bonus: {mean_bonus:.2f} | Gate: {gate_mean:.2f} | Value Loss: {total_v_loss / (PPO_EPOCHS * NUM_MINIBATCHES):.3f}")
            
            # Robust Checkpointing
            if update % 50 == 0: # Every ~200k steps
                save_path = f"{SAVE_DIR}/inn_crafter_{global_step}.pt"
                torch.save({
                    'agent_state_dict': agent.state_dict(),
                    'optimizer_actor_state_dict': opt_actor.state_dict(),
                    'optimizer_critic_state_dict': opt_critic.state_dict(),
                    'global_step': global_step,
                }, save_path)
                print(f"💾 Checkpoint saved: {save_path}")

    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user. Saving emergency checkpoint...")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        final_save_path = f"{SAVE_DIR}/inn_crafter_final.pt"
        torch.save({
            'agent_state_dict': agent.state_dict(),
            'optimizer_actor_state_dict': opt_actor.state_dict(),
            'optimizer_critic_state_dict': opt_critic.state_dict(),
            'global_step': global_step,
        }, final_save_path)
        print(f"✅ Final/Emergency Checkpoint saved: {final_save_path}")
        envs.close()