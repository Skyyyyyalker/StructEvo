import logging
import torch
import pandas as pd
import numpy as np
from typing import List
from Levenshtein import distance as levenshtein
from omegaconf import OmegaConf
from structevo.reward.utils import Encoder, BaseCNN

AMINO_ACIDS = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

def get_logger(name) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter=logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    
    handler1=logging.StreamHandler()
    handler1.setLevel(logging.DEBUG)
    handler1.setFormatter(formatter)
    logger.addHandler(handler1)
    return logger

MASKED_SEQ = {
    "GB1": "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNG___EWTYDDATKTFT_TE",
    "PhoQ": "SYMVWSWFIYVLSANLLLVIPLLWVAAWWSLRPIEALAKEVRELEEHNRELLNPATTRELTSLVRNLNRLLKSERERYDKYRTTLTDLTHSLKTPL__LQ__LRSLRSEKMSVSDAEPVMLEQISRISQQIGYYLHRASMRGGTLLSRELHPVAPLLDNLTSALNKVYQRKGVNISLDISPEISFVGEQ"
}

MASKED_INDICES = {
    "GB1": [38, 39, 40, 53],
    "PhoQ": [96, 97, 100, 101],
}

WILDTYPE = {
    "GB1": "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",
    "PhoQ": "SYMVWSWFIYVLSANLLLVIPLLWVAAWWSLRPIEALAKEVRELEEHNRELLNPATTRELTSLVRNLNRLLKSERERYDKYRTTLTDLTHSLKTPLAVLQSTLRSLRSEKMSVSDAEPVMLEQISRISQQIGYYLHRASMRGGTLLSRELHPVAPLLDNLTSALNKVYQRKGVNISLDISPEISFVGEQ",
    'AAV': 'DEEEIKATNPVATERFGTVAVNFQSSST',
    'GFP': 'SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKRHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK'
}

REFSEQ = {
    'AAV': {
        'medium': 'DEEEIRTTNPVATEQYGSVETPDEVGNC',
        'hard': 'DEEEIRTTNPFATEQYGSVEEGECQGDF'
    },
    'GFP': {
        'medium': 'SKGEELFTGVVPILVELDGDVNGHKSSVSGEGEGDATYGKLTLKFICTTGKLPVPRPTLATTLSYGVQCLSRYPDHMRQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVSFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK',
        'hard': 'SKGEELFTGVVPILVELDGDVDGHKFSVSGEGEGDATYGKLTLKSICTTGKLPVPWPALVTTLSYGVQCFSRYPDHMKQHDFFKSAMPVGYVQERTIFLKDDGNYKTRAEVRFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEGGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK'
    }
}


class OracleRunner:
    def __init__(self, oracle_path: str, device: torch.device):
        self.device = device
        oracle_cfg_path = oracle_path.replace('.ckpt', '_config.yaml')
        with open(oracle_cfg_path, 'r') as fp:
            ckpt_cfg = OmegaConf.load(fp.name)
        self.oracle_model = BaseCNN(**ckpt_cfg.model.predictor)
        oracle_state_dict = torch.load(oracle_path, map_location="cpu")
        self.oracle_model.load_state_dict(
            {k.replace('predictor.', ''): v for k,v in oracle_state_dict['state_dict'].items()})
        self.oracle_model = self.oracle_model.to(self.device)
        self.oracle_model.eval()
    
        self.predictor_tokenizer = Encoder()
        self.batch_size = 32

    def tokenize(self, seqs):
        return self.predictor_tokenizer.encode(seqs).to(self.device)
    
    @torch.no_grad()
    def get_oracle_scores(self, seqs):
        """ get oracle scores for sequences """
        tokenized_seqs = self.tokenize(seqs)
        batches = torch.split(tokenized_seqs, self.batch_size, 0)
        scores = []
        for b in batches:
            if b is None: continue
            results = self.oracle_model(b).detach()
            scores.append(results)
        return torch.concat(scores, dim=0).cpu().numpy().tolist()


class EvalRunner:
    def __init__(self, gt_path, initpool_path, use_normalization=True):
        self._log = get_logger("EvalRunner")
        self.use_normalization = use_normalization

        # Read groundtruth csv: used for max-min normalization
        df_gt = pd.read_csv(gt_path)
        self._max_gt_score = np.max(df_gt["GroundTruth"])
        self._min_gt_score = np.min(df_gt["GroundTruth"])
        self._log.info(f'Read in {len(df_gt)} ground truth sequences.')
        self._log.info(f'Maximum gt score: {self._max_gt_score}.')
        self._log.info(f'Minimum gt score: {self._min_gt_score}.')

        # Read initpool csv: used for calculate d_init
        df_initpool = pd.read_csv(initpool_path)
        self._base_pool_seqs = df_initpool["Sequence"].tolist()
        self._log.info(f'Read in {len(self._base_pool_seqs)} base pool sequences.')
        self._log.info(f'Maximum initpool score: {df_initpool["GroundTruth"].max()}.')
        self._log.info(f'Minimum initpool score: {df_initpool["GroundTruth"].min()}.')

    def _normalize(self, x: float) -> float:
        return (x - self._min_gt_score) / (self._max_gt_score - self._min_gt_score)

    def _cal_diversity(self, seqs) -> float:
        """ mean inner distance """
        num_seqs = len(seqs)
        total_dist = 0
        for i in range(num_seqs):
            for j in range(num_seqs):
                x = seqs[i]
                y = seqs[j]
                if x == y:
                    continue
                total_dist += levenshtein(x, y)
        return total_dist / (num_seqs*(num_seqs-1))

    def _cal_novelty(self, seqs) -> List:
        all_novelty = []
        for src in seqs:  
            min_dist = 1e9
            for known in self._base_pool_seqs:
                dist = levenshtein(src, known)
                if dist < min_dist:
                    min_dist = dist
            all_novelty.append(min_dist)
        return all_novelty

    def evaluate_sequences(self, df_topk: pd.DataFrame) -> pd.DataFrame:
        """ 
        Inputs: Dataframe with columns: `Sequence`, `GroundTruth`
        Return: a one-line Dataframe
        """
        topk_seqs = df_topk["Sequence"].tolist()
        num_unique_seqs = len(list(set(topk_seqs)))
        seq_novelty = self._cal_novelty(topk_seqs)
        seq_diversity = self._cal_diversity(topk_seqs)
        
        topk_scores = df_topk["GroundTruth"].tolist()
        if self.use_normalization:
            scores = [self._normalize(x) for x in topk_scores]
        else:
            scores = topk_scores

        result_df = pd.DataFrame({
            'num_unique': [num_unique_seqs],
            'mean_fitness': [np.mean(scores)],
            'median_fitness': [np.median(scores)],
            'max_fitness': [np.max(scores)],
            'mean_diversity': [seq_diversity],
            'mean_novelty': [np.mean(seq_novelty)],
            'median_novelty': [np.median(seq_novelty)],
        })
        return result_df