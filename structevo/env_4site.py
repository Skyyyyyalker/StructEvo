import argparse
import os
import torch
import random
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from transformers import EsmTokenizer

from structevo.geo_ppo import PPOwithGeoLoss
from structevo.policy_4site import MutationPolicy
from structevo.utils import AMINO_ACIDS, MASKED_SEQ, MASKED_INDICES, get_logger
log = get_logger("Train PPO")

collected_combo_dict = {}  # key: Seq(str); value: PredReward

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['GB1', "PhoQ"])
    parser.add_argument('--round', type=int, required=True, help='current round for RL, starts at 1')
    parser.add_argument('--n_candidates', type=int, default=96, help='proposed top candidates sorted by proxy')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0')
    # ppo args
    parser.add_argument('--n_envs', type=int, default=8, help='Number of parallel environments')
    parser.add_argument('--total_steps', type=int, default=3000, help="Total steps in single env")
    parser.add_argument('--max_steps', type=int, default=3, help='Maxium steps in one episode')
    parser.add_argument('--n_steps', type=int, default=256, help='Number of collected steps in single env for policy update')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for policy update')
    parser.add_argument('--clip', type=float, default=0.3, help="ppo clip range")
    parser.add_argument('--ent_coef', type=float, default=0.0, help="Entropy coefficient for encouraging exploration")
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate for policy update')
    parser.add_argument('--gamma', type=float, default=0.99, help="discount_factor")
    # path args
    parser.add_argument('--seq_encoder_path', type=str, default="./ckpts/esm2_t33_650M_UR50D")
    parser.add_argument('--proxy_ckpt', type=str, default=None)
    parser.add_argument('--ground_truth_path', type=str, default=None)
    parser.add_argument('--init_sequence_path', type=str, default=None)
    parser.add_argument('--candidates_save_path', type=str, default=None)
    parser.add_argument('--structure_filepath', type=str, default=None)
    parser.add_argument('--structure_chain_ids', type=str, default='A')
    parser.add_argument('--tensorboard_logdir', type=str, default=None)
    parser.add_argument('--tensorboard_logname', type=str, default=None)
    return parser.parse_args()


class LoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(LoggingCallback, self).__init__(verbose)
        
    def _on_step(self) -> bool:
        # will be called after each call to env.step(). record from returned `info` dict
        if "infos" in self.locals:
            for info in self.locals["infos"]:
                if info and "state/reward" in info:
                    self.logger.record("state/reward", info["state/reward"])
                if info and "state/n_proposed" in info:
                    self.logger.record("state/n_proposed", info["state/n_proposed"])
        return True


class MutaEnv(gym.Env):
    def __init__(self, args: dict):
        super().__init__()
        df_init = pd.read_csv(args.init_sequence_path)
        self.init_combo_pool = df_init['AACombo'].tolist()
        self.masked_seq = MASKED_SEQ[args.task]
        self.mask_indices = MASKED_INDICES[args.task]
        self.state_len = len(self.init_combo_pool[0])   # 4
        self.protein_len = len(self.masked_seq)
        log.info(
            f"number of init sequences: {len(self.init_combo_pool)}, "
            f"state length: {self.state_len}, "
            f"protein length: {self.protein_len}"
        )

        self.space_dim = len(AMINO_ACIDS)
        self.action_space = spaces.MultiDiscrete([self.state_len, self.space_dim])
        self.observation_space = spaces.MultiDiscrete([self.space_dim] * self.state_len)
        self.tokenizer = EsmTokenizer.from_pretrained(args.seq_encoder_path)
        
        self.curr_step = 0
        self.stop_criteria = -1
        self.device = args.device
        self.max_steps = args.max_steps
        self._init_proxy(args)

    def _init_proxy(self, args):
        log.info(f"Loading proxy (csv file) from {args.proxy_ckpt}")
        assert args.proxy_ckpt.endswith('.csv')
        df = pd.read_csv(args.proxy_ckpt)
        self.proxy = {k: v for k, v in zip(df['AACombo'], df['PredictedFitness'])}

    def combo2seq(self, combo: str) -> str:
        return self.masked_seq.replace("_", "{}").format(*combo)

    def seq2combo(self, sequence: str) -> str:
        return ''.join([sequence[i] for i in self.mask_indices])
    
    def combo2state(self, combo: str) -> np.ndarray:
        token = np.array(self.tokenizer(combo, add_special_tokens = False).input_ids)
        return token - 4
    
    def state2combo(self, state: np.ndarray) -> str:
        token = state + 4
        return ''.join(self.tokenizer.decode(token).split())
    
    def state2seq(self, state: np.ndarray) -> str:
        return self.combo2seq(self.state2combo(state))

    def seq2state(self, sequence: str) -> np.ndarray:
        return self.combo2state(self.seq2combo(sequence))
    
    def reset(self, seed=None, options=None) -> np.ndarray:
        super().reset(seed=seed)
        self.seed = seed
        random.seed(seed)
        
        self.init_combo = random.choice(self.init_combo_pool)
        self.init_state = self.combo2state(self.init_combo)
        self.state = self.init_state.copy()
        self.curr_step = 0
        reward = self._get_reward(self.state)
        self.stop_criteria = reward
        info = {
            'init_combo': self.init_combo,
            'init_state': self.init_state,
            'init_reward': reward
        }
        collected_combo_dict[self.init_combo] = reward
        return self.state, info

    def _check_done(self, curr_reward):
        terminated, truncated = False, False
        if curr_reward > self.stop_criteria:   
            terminated = True
        if self.curr_step >= self.max_steps:
            truncated = True
        return terminated, truncated
    
    def _get_reward(self, state: np.ndarray) -> float:
        return self.proxy[self.state2combo(state)]

    def _get_new_state(self, state: np.ndarray, action: torch.Tensor):
        pos, new_aa = action
        state[pos] = new_aa
        return state
    
    def step(self, action):
        self.curr_step += 1
        new_state = self._get_new_state(self.state, action)
        old_combo = self.state2combo(self.state)
        new_combo = self.state2combo(new_state)
        reward = self._get_reward(new_state)
        terminated, truncated = self._check_done(reward)
        if terminated or truncated:
            collected_combo_dict[new_combo] = reward
        else:   # no reward for intermediate steps
            reward = 0.0

        info = {
            'curr_step': self.curr_step,
            'terminated': terminated,
            'truncated': truncated,
            'action': action,
            'old_combo': old_combo,
            'new_combo': new_combo,
            'init_combo': self.init_combo,
            'rewards': reward
        }
        self.state = new_state
        return self.state, reward, terminated, truncated, info


def print_trainable_parameters(model: torch.nn.Module):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    log.info(f"Trainable params: {trainable_params} || "
            f"all params: {all_param} || "
            f"trainable%: {100 * trainable_params / all_param}")


def get_env(args):
    log.info(f"Creating {args.n_envs} parallel environment...")
    env = make_vec_env(
        MutaEnv, 
        n_envs=args.n_envs,
        seed=args.seed,
        env_kwargs={"args": args}
    )
    return env


def get_ppo_model(env, args):
    log.info("Loading PPO model...")
    model = PPOwithGeoLoss(
        policy=MutationPolicy,
        env=env,
        gamma=args.gamma, 
        clip_range=args.clip, 
        ent_coef=args.ent_coef, 
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        tensorboard_log=args.tensorboard_logdir, 
        verbose=1, 
        device=args.device,
        policy_kwargs=dict(
            masked_seq=MASKED_SEQ[args.task],
            masked_indices=MASKED_INDICES[args.task],
            seq_encoder_path=args.seq_encoder_path,
            structure_filepath=args.structure_filepath,
            chain_ids=args.structure_chain_ids.split(",")
        ),
    )
    print_trainable_parameters(model.policy)
    return model


def train(model, args):
    if not os.path.exists(os.path.dirname(args.tensorboard_logdir)):
        log.info(f"Creating log saving path: {os.path.dirname(args.tensorboard_logdir)}")
        os.makedirs(os.path.dirname(args.tensorboard_logdir))
    log.info("Start training PPO!")
    model.learn(
        total_timesteps=args.total_steps,
        tb_log_name=args.tensorboard_logname,
        callback=LoggingCallback()
    )
    log.info("Finish training PPO.")


def save_collected_sequences(args):
    if not os.path.exists(os.path.dirname(args.candidates_save_path)):
        log.info(f"Creating candidates save path: {os.path.dirname(args.candidates_save_path)}")
        os.makedirs(os.path.dirname(args.candidates_save_path))
    
    df_init = pd.read_csv(args.init_sequence_path).drop(columns=["Unnamed: 0"])
    init_combo_pool = df_init['AACombo'].tolist()
    masked_seq = MASKED_SEQ[args.task]
    
    # save all collected sequences
    combos = list(collected_combo_dict.keys())
    rewards = list(collected_combo_dict.values())
    intrainings = [1 if combo in init_combo_pool else 0 for combo in combos]
    df = pd.DataFrame({
        "AACombo": combos,
        "PredReward": rewards,
        "InTrainingData": intrainings
    })
    sorted_df = df.sort_values(by="PredReward", ascending=False).reset_index(drop=True)
    unprocessed_path = args.candidates_save_path.replace('.csv', '_allproposed.csv')
    sorted_df.to_csv(unprocessed_path)
    log.info(f"Collected {len(collected_combo_dict)} unique unselected sequences in round {args.round}. Saved to {unprocessed_path}")
    
    df_gt = pd.read_csv(args.ground_truth_path)
    gt_combos = set(df_gt['AACombo'].tolist())
    
    # select top new candidates
    top96_combos = []
    top96_seqs = []
    top96_predrewards = []
    top96_groundtruths = []
    for i in range(len(sorted_df)):
        if sorted_df["InTrainingData"][i] == 0:
            combo = sorted_df["AACombo"][i]
            seq = masked_seq.replace("_", "{}").format(*combo)
            predrewards = sorted_df["PredReward"][i]
            if combo in gt_combos:
                groundtruth = df_gt[df_gt["AACombo"] == combo]["GroundTruth"].values[0]
            else:
                groundtruth = 0.0    # missing ground truth

            top96_combos.append(combo)
            top96_seqs.append(seq)
            top96_predrewards.append(predrewards)
            top96_groundtruths.append(groundtruth)
        
        if len(top96_combos) >= args.n_candidates:
            break
    
    assert len(top96_combos) == args.n_candidates, \
        f"Not enough new candidates: Collected {len(top96_combos)}, expected {args.n_candidates}."

    df_new96 = pd.DataFrame({
        "AACombo": top96_combos,
        "Sequence": top96_seqs,
        "PredReward": top96_predrewards,
        "GroundTruth": top96_groundtruths
    })
    df_all = pd.concat([df_init, df_new96], axis=0).reset_index(drop=True)
    df_all.to_csv(args.candidates_save_path)
    log.info(f"Successfully saved {len(df_all)} candidates at {args.candidates_save_path}")


def main(args):
    log.info(args)
    env = get_env(args)
    model = get_ppo_model(env, args)
    train(model, args)
    save_collected_sequences(args)


if __name__ == '__main__':
    args = get_args()
    main(args)
