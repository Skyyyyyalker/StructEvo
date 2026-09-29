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
from Levenshtein import distance as levenshtein

from structevo.geo_ppo import PPOwithGeoLoss
from structevo.policy_full import MutationPolicy
from structevo.reward.utils import BaseCNN, seq_to_one_hot
from structevo.utils import AMINO_ACIDS, REFSEQ, OracleRunner, get_logger
log = get_logger("Train PPO")

collected_seq_dict = {}  # key: Seq(str); value: PredReward

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['GFP-medium', 'GFP-hard', 'AAV-medium', 'AAV-hard'])
    parser.add_argument('--round', type=int, required=True, help='current round for RL, starts at 1')
    parser.add_argument('--n_candidates', type=int, default=256, help='proposed top candidates sorted by proxy')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0')
    # ppo args
    parser.add_argument('--n_envs', type=int, default=8, help='Number of parallel environments')
    parser.add_argument('--total_steps', type=int, default=15000, help="Total steps in single env")
    parser.add_argument('--max_steps', type=int, default=3, help='Maxium steps in one episode')
    parser.add_argument('--n_steps', type=int, default=256, help='Number of collected steps in single env for policy update')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for policy update')
    parser.add_argument('--clip', type=float, default=0.3, help="ppo clip range")
    parser.add_argument('--ent_coef', type=float, default=0.0, help="Entropy coefficient for encouraging exploration")
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate for policy update')
    parser.add_argument('--gamma', type=float, default=0.99, help="discount_factor")
    # path args
    parser.add_argument('--seq_encoder_path', type=str, default="./ckpts/esm2_t6_8M_UR50D")
    parser.add_argument('--proxy_ckpt', type=str, default=None)
    parser.add_argument('--oracle_ckpt', type=str, default=None)
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
        self.init_pool = self._get_initseqs(args.init_sequence_path, args.round)
        self.state_len = len(self.init_pool[0])
        self.space_dim = len(AMINO_ACIDS)
        self.action_space = spaces.MultiDiscrete([self.state_len, self.space_dim])
        self.observation_space = spaces.MultiDiscrete([self.space_dim] * self.state_len)
        self.tokenizer = EsmTokenizer.from_pretrained(args.seq_encoder_path)
        log.info(
            f"Number of init sequences: {len(self.init_pool)}, "
            f"state length: {self.state_len}, "
        )
        
        self.curr_step = 0
        self.stop_criteria = -1
        self.device = args.device
        self.max_steps = args.max_steps
        protein, level = args.task.split("-")
        self.ref_seq = REFSEQ[protein][level]
        self._init_proxy(args)

    def _get_initseqs(self, initpool_path, curr_round):
        # get top 128 seqs from candidates pool
        candidates_pool = [initpool_path.replace(f"round_{curr_round-1}", f"round_{i}") for i in range(curr_round)]
        df_init = [pd.read_csv(f) for f in candidates_pool]
        df_init = pd.concat(df_init, ignore_index=True).reset_index()
        df_init = df_init.sort_values(by='GroundTruth', ascending=False).reset_index(drop=True)
        df_init = df_init.iloc[:128]
        return df_init['Sequence'].tolist()

    def _init_proxy(self, args):
        log.info(f"Loading proxy from {args.proxy_ckpt}")
        proxy = BaseCNN(make_one_hot=False)
        predictor_ckpt = torch.load(args.proxy_ckpt, map_location=self.device)
        if "state_dict" in predictor_ckpt.keys():
            predictor_ckpt = predictor_ckpt["state_dict"]
        predictor_ckpt = {k.replace('model.',''):v for k,v in predictor_ckpt.items()}
        predictor_ckpt = {k.replace('predictor.',''):v for k,v in predictor_ckpt.items()}
        proxy.load_state_dict(predictor_ckpt)
        self.proxy = proxy.to(self.device)
        self.proxy.eval()

    def seq2state(self, seq: str) -> np.ndarray:
        # state_id = esm_token_id - 4
        token = np.array(self.tokenizer(seq, add_special_tokens = False).input_ids)
        return token - 4
    
    def state2seq(self, state: np.ndarray) -> str:
        token = state + 4
        return ''.join(self.tokenizer.decode(token).split())
    
    def reset(self, seed=None, options=None) -> np.ndarray:
        super().reset(seed=seed)
        self.seed = seed
        random.seed(seed)
        
        self.init_seq = random.choice(self.init_pool)
        self.init_state = self.seq2state(self.init_seq)
        self.state = self.init_state.copy()
        self.curr_step = 0
        reward = self._get_reward(self.state)
        self.stop_criteria = reward
        info = {
            'init_seq': self.init_seq,
            'init_state': self.init_state,
            'init_reward': reward
        }
        collected_seq_dict[self.init_seq] = reward
        return self.state, info

    def _distance_to_ref(self, seq: str):
        return levenshtein(seq, self.ref_seq)

    def _check_done(self, curr_reward, new_seq):
        terminated, truncated = False, False
        if curr_reward > self.stop_criteria:   
            terminated = True
        if self.curr_step >= self.max_steps or self._distance_to_ref(new_seq) >= 15:
            truncated = True
        return terminated, truncated
    
    def _get_reward(self, state: np.ndarray) -> float:
        seq = self.state2seq(state)
        with torch.no_grad():
            tokenized_seq = seq_to_one_hot(seq).to(self.device)
            return self.proxy(tokenized_seq.unsqueeze(0)).item()

    def _get_new_state(self, state: np.ndarray, action: torch.Tensor):
        pos, new_aa = action
        state[pos] = new_aa
        return state
    
    def step(self, action):
        self.curr_step += 1
        new_state = self._get_new_state(self.state, action)
        old_seq = self.state2seq(self.state)
        new_seq = self.state2seq(new_state)
        reward = self._get_reward(new_state)
        terminated, truncated = self._check_done(reward, new_seq)
        if terminated or truncated:
            collected_seq_dict[new_seq] = reward
        else:   # no reward for intermediate steps
            reward = 0.0

        info = {
            'curr_step': self.curr_step,
            'terminated': terminated,
            'truncated': truncated,
            'action': action,
            'old_seq': old_seq,
            'new_seq': new_seq,
            'init_seq': self.init_seq,
            'state/reward': reward,
            'state/n_proposed': len(collected_seq_dict)
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
    all_candidates = [args.init_sequence_path.replace(f"round_{args.round-1}", f"round_{i}") for i in range(args.round)]
    df_init = [pd.read_csv(f) for f in all_candidates]
    df_init = pd.concat(df_init, ignore_index=True).reset_index()
    init_seq_pool = df_init['Sequence'].tolist()
    log.info(f"Total number of candidate sequences until now: {len(init_seq_pool)}")
    
    # save all collected sequences
    seqs = list(collected_seq_dict.keys())
    rewards = list(collected_seq_dict.values())
    intrainings = [1 if seq in init_seq_pool else 0 for seq in seqs]
    df = pd.DataFrame({
        "Sequence": seqs,
        "PredReward": rewards,
        "InTrainingData": intrainings
    })
    sorted_df = df.sort_values(by="PredReward", ascending=False).reset_index(drop=True)
    unprocessed_path = args.candidates_save_path.replace('.csv', '_allproposed.csv')
    sorted_df.to_csv(unprocessed_path)
    log.info(f"Collected {len(collected_seq_dict)} unique unselected sequences in round {args.round}. Saved to {unprocessed_path}")
    
    # select top new candidates
    candidate_seqs = []
    candidate_predrewards = []
    candidate_groundtruths = []
    for i in range(len(sorted_df)):
        if sorted_df["InTrainingData"][i] == 0:
            candidate_seqs.append(sorted_df["Sequence"][i])
            candidate_predrewards.append(sorted_df["PredReward"][i])
        if len(candidate_seqs) >= args.n_candidates:
            break
    assert len(candidate_seqs) == args.n_candidates, \
        f"Not enough new candidates: Collected {len(candidate_seqs)}, expected {args.n_candidates}."
    
    oracle_runner = OracleRunner(args.oracle_ckpt, args.device)
    candidate_groundtruths = oracle_runner.get_oracle_scores(candidate_seqs)
    df_candidates = pd.DataFrame({
        "Sequence": candidate_seqs,
        "PredReward": candidate_predrewards,
        "GroundTruth": candidate_groundtruths
    })
    df_candidates.to_csv(args.candidates_save_path)
    log.info(f"Successfully saved {len(df_candidates)} candidates at {args.candidates_save_path}")


def main(args):
    log.info(args)
    env = get_env(args)
    model = get_ppo_model(env, args)
    train(model, args)
    save_collected_sequences(args)


if __name__ == '__main__':
    args = get_args()
    main(args)
