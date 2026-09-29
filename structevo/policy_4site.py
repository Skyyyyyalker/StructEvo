from typing import Optional, Union, Any
import torch
from torch import nn

from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.distributions import CategoricalDistribution
from transformers import EsmModel, EsmTokenizer
import esm


class SeqFeatureExtractor(nn.Module):
    """ Sequence-only feature Extractor """
    def __init__(self, 
        observation_space, 
        seq_encoder_path,
        masked_seq,
    ):
        super().__init__()
        self.observation_space = observation_space
        self.pos_space_length = observation_space.shape[0]
        self.seq_encoder_path = seq_encoder_path
        self.masked_seq = masked_seq
        self._load_seq_encoder()

    def _load_seq_encoder(self):
        self.seq_encoder = EsmModel.from_pretrained(self.seq_encoder_path)
        self.seq_tokenizer = EsmTokenizer.from_pretrained(self.seq_encoder_path)
        self.seq_features_dim = self.seq_encoder.config.hidden_size
        self.features_dim = self.seq_features_dim
        for p in self.seq_encoder.parameters():
            p.requires_grad = False
        self.seq_encoder.eval()

    def state2seq(self, obs: torch.Tensor) -> torch.Tensor:
        token = obs + 4
        combo = ''.join(self.seq_tokenizer.decode(token).split())
        seq = self.masked_seq.replace("_", "{}").format(*combo)
        return seq
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        inputs: state [B, pos_space_len]
        return: condition-modulated sequence embedding [B, binder_len, seq_hidden_dim]
        """
        batchsize = obs.shape[0]
        device = obs.device
        seqs = [self.state2seq(obs[i]) for i in range(batchsize)]
        esm_inputs = self.seq_tokenizer(
            seqs,
            add_special_tokens=False,
            return_tensors='pt'
        ).to(device)
        with torch.no_grad():
            seq_feature = self.seq_encoder(**esm_inputs).last_hidden_state
        return seq_feature


class CrossattnFeatureExtractor(nn.Module):
    def __init__(self, 
        observation_space, 
        seq_encoder_path, 
        masked_seq,
        struct_filepath,
        chain_ids=["A"],
    ):
        super().__init__()
        self.observation_space = observation_space
        self.pos_space_length = observation_space.shape[0]
        self.seq_encoder_path = seq_encoder_path
        self.struct_filepath = struct_filepath
        self.chain_ids = chain_ids
        self.masked_seq = masked_seq

        self._load_seq_encoder()
        self._load_structure_encoder()
        self.struct_proj = nn.Linear(self.struct_features_dim, self.seq_features_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=self.seq_features_dim, num_heads=8, batch_first=True)

    def _load_seq_encoder(self):
        self.seq_encoder = EsmModel.from_pretrained(self.seq_encoder_path)
        self.seq_tokenizer = EsmTokenizer.from_pretrained(self.seq_encoder_path)
        self.seq_features_dim = self.seq_encoder.config.hidden_size
        self.features_dim = self.seq_features_dim
        for p in self.seq_encoder.parameters():
            p.requires_grad = False
        self.seq_encoder.eval()

    def _load_structure_encoder(self):
        struct_encoder, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        self.struct_features_dim = struct_encoder.encoder.args.encoder_embed_dim
        for p in struct_encoder.parameters():
            p.requires_grad = False
        struct_encoder.eval()

        # get structure embedding
        structure = esm.inverse_folding.util.load_structure(self.struct_filepath, self.chain_ids)
        coords, _ = esm.inverse_folding.multichain_util.extract_coords_from_complex(structure)
        with torch.no_grad():
            self.struct_emb = torch.cat([
                esm.inverse_folding.multichain_util.get_encoder_output_for_complex(struct_encoder, alphabet, coords, chain_id) 
                for chain_id in self.chain_ids
            ], dim=0)

    def state2seq(self, obs: torch.Tensor) -> torch.Tensor:
        token = obs + 4
        combo = ''.join(self.seq_tokenizer.decode(token).split())
        seq = self.masked_seq.replace("_", "{}").format(*combo)
        return seq
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        inputs: state [B, pos_space_len]
        return: condition-modulated sequence embedding [B, binder_len, seq_hidden_dim]
        """
        batchsize = obs.shape[0]
        device = obs.device
        # seq
        seqs = [self.state2seq(obs[i]) for i in range(batchsize)]
        esm_inputs = self.seq_tokenizer(
            seqs,
            add_special_tokens=False,
            return_tensors='pt'
        ).to(device)
        with torch.no_grad():
            seq_feature = self.seq_encoder(**esm_inputs).last_hidden_state
        # structure
        struct_emb_expanded = self.struct_emb.expand(batchsize, -1, -1).to(device)
        struct_feature = self.struct_proj(struct_emb_expanded)   # [B, 230, 320]
        # fusion
        fusion_feature = self.cross_attn(seq_feature, struct_feature, struct_feature)[0]
        return fusion_feature


class PosActionNet(nn.Module):
    """ 
    inputs: state logits; 
    return: position logits
    """
    def __init__(self, feature_hidden_dim, pos_space_len):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.pos_head = nn.Sequential(
            nn.Linear(feature_hidden_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, pos_space_len)
        )
        self._initialize_weights(0.01)
    
    def _initialize_weights(self, gain=0.01):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=gain)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)  
    
    def forward(self, prot_latent: torch.Tensor):
        prot_latent = prot_latent.permute(0, 2, 1)
        prot_latent = self.pool(prot_latent).squeeze(-1)
        pos_logits = self.pos_head(prot_latent)     # [batchsize, pos_space_len]
        return pos_logits


class AAtypeActionNet(nn.Module):
    """ 
    inputs: state logits; position
    return: action logits
    """
    def __init__(self, 
                 feature_hidden_dim, 
                 num_aas = 20,
                 masked_indices = None):
        super().__init__()
        self.aa_head = nn.Sequential(
            nn.Linear(feature_hidden_dim, num_aas),
            # nn.ReLU(),
            # nn.LayerNorm(128),
            # nn.Linear(128, num_aas)
        )
        self._initialize_weights(0.01)
        self.pos2actual = {i: actual_pos for i, actual_pos in enumerate(masked_indices)}
    
    def _initialize_weights(self, gain=0.01):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=gain)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)  
    
    def forward(self, h: torch.Tensor, pos: torch.LongTensor, aa_orig: torch.LongTensor):
        batchsize = h.shape[0]
        actual_pos = torch.tensor([self.pos2actual[p.item()] for p in pos], device=pos.device)
        aa_logits_at_pos = h[torch.arange(batchsize), actual_pos, :].detach()
        aa_type_logits = self.aa_head(aa_logits_at_pos)
        aa_type_logits[torch.arange(batchsize), aa_orig] = -1e9
        return aa_type_logits


class ValueNet(nn.Module):
    """ 
    Inputs: [bs, prot_len, hidden_dim]; 
    Return: V(state), [bs, 1]
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 1)
        )
        self._initialize_weights(1)
        
    def _initialize_weights(self, gain=0.01):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=gain)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    
    def forward(self, latent_per_res: torch.Tensor):
        latent_seq = latent_per_res.mean(dim=1)
        value_seq = self.value_net(latent_seq)
        return value_seq


class MutationPolicy(BasePolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        use_sde: bool = False,
        activation_fn: type[nn.Module] = nn.Tanh,
        optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        seq_encoder_path: str = None,
        structure_filepath: str = None,
        chain_ids: list = ["A"],
        masked_seq: str = None,
        masked_indices: list = None,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            if optimizer_class == torch.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super().__init__(
            observation_space,
            action_space,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )
        self.activation_fn = activation_fn
        self.use_sde = use_sde
        self.lr_schedule = lr_schedule
        self.structure_filepath = structure_filepath
        self.chain_ids = chain_ids if chain_ids != [''] else []
        self.seq_encoder_path = seq_encoder_path
        self.masked_seq = masked_seq
        self.masked_indices = masked_indices
        self._build()

    def _build(self) -> None:
        # sequence-only feature extractor
        # self.features_extractor = SeqFeatureExtractor(
        #     self.observation_space,
        #     self.seq_encoder_path
        # )
        self.features_extractor = CrossattnFeatureExtractor(
            self.observation_space,
            self.seq_encoder_path,
            self.masked_seq,
            self.structure_filepath, 
            self.chain_ids
        )
        
        # build action net
        self.action_dims = self.action_space.nvec.tolist()  # [4, 20]
        self.pos_space_dim, self.aa_space_dim = self.action_dims
        self.pos_action_dist = CategoricalDistribution(self.pos_space_dim)
        self.aa_action_dist = CategoricalDistribution(self.aa_space_dim)
        self.pos_action_net = PosActionNet(
            self.features_extractor.features_dim, 
            self.pos_space_dim
        )
        self.aa_action_net = AAtypeActionNet(
            self.features_extractor.features_dim, 
            self.aa_space_dim,
            self.masked_indices
        )
        
        # build value net
        self.value_net = ValueNet(self.features_extractor.features_dim)
        
        # build optimizer
        self.optimizer = self.optimizer_class(
            self.parameters(), 
            lr=self.lr_schedule(1), 
            **self.optimizer_kwargs
        )

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation (each aa in aaid format)
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        features = self.extract_features(obs)       # features: [bs, seq_len, hidden_dim]
        
        # Step 1. get pos_dist for sampling pos & getting pos_log_prob
        pos_logits = self.pos_action_net(features)
        pos_dist = self.pos_action_dist.proba_distribution(action_logits=pos_logits)
        pos_actions = pos_dist.get_actions(deterministic=deterministic)
        pos_log_probs = pos_dist.log_prob(pos_actions)               # [bs]
        
        # Step 2. Given pos, get aa_dist for sampling aa & getting aa_log_prob
        ori_aas = obs[torch.arange(obs.size(0)), pos_actions]
        aa_logits = self.aa_action_net(features, pos_actions, ori_aas)
        aa_dist = self.aa_action_dist.proba_distribution(action_logits=aa_logits)
        aa_actions = aa_dist.get_actions(deterministic=deterministic)
        aa_log_probs = aa_dist.log_prob(aa_actions)
        
        # Step 3. action = (pos, aa), log_probs = pos_log_prob + aa_log_prob
        actions = torch.stack([pos_actions, aa_actions], dim=1)
        log_probs = pos_log_probs + aa_log_probs
        
        values = self.value_net(features)
        return actions, values, log_probs


    def extract_features(self, obs: torch.Tensor) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        return self.features_extractor(obs)


    def _predict(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        features = self.extract_features(obs)
        
        # Step 1. get pos_dist for sampling pos & getting pos_log_prob
        pos_logits = self.pos_action_net(features)
        pos_dist = self.pos_action_dist.proba_distribution(action_logits=pos_logits)
        pos_actions = pos_dist.get_actions(deterministic=deterministic)
        
        # Step 2. Given pos, get aa_dist for sampling aa & getting aa_log_prob
        ori_aas = obs[torch.arange(obs.size(0)), pos_actions]
        aa_logits = self.aa_action_net(features, pos_actions, ori_aas)
        aa_dist = self.aa_action_dist.proba_distribution(action_logits=aa_logits)
        aa_actions = aa_dist.get_actions(deterministic=deterministic)
        
        # Step 3. action = [(pos, aa)]
        actions = torch.stack([pos_actions, aa_actions], dim=1)
        return actions

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        :return: 
            estimated value, 
            log likelihood of taking those actions, and 
            entropy of the action distribution.
        """
        obs = obs.to(dtype=torch.long)
        pos_actions = actions[:, 0]
        aa_actions = actions[:, 1]
        features = self.extract_features(obs)            # [bs, seq_len, hidden_dim]
        
        # Step 1. get pos_dist for sampling pos & getting pos_log_prob
        pos_logits = self.pos_action_net(features)
        pos_dist = self.pos_action_dist.proba_distribution(action_logits=pos_logits)
        pos_log_probs = pos_dist.log_prob(pos_actions)
        pos_ents = pos_dist.entropy()
        
        # Step 2. Given pos, get aa_dist for sampling aa & getting aa_log_prob
        pos_actions = pos_actions.to(dtype=torch.long)
        ori_aas = obs[torch.arange(obs.size(0)), pos_actions]
        aa_logits = self.aa_action_net(features, pos_actions, ori_aas)
        aa_dist = self.aa_action_dist.proba_distribution(action_logits=aa_logits)
        aa_log_probs = aa_dist.log_prob(aa_actions)
        aa_ents = aa_dist.entropy()
        
        # Step 3. log_probs = pos_log_prob + aa_log_prob
        log_probs = pos_log_probs + aa_log_probs
        entropy = pos_ents + aa_ents
        
        values = self.value_net(features)
        return values, log_probs, entropy


    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(obs)
        return self.value_net(features)
