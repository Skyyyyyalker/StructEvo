import os
import shutil
import argparse
import pandas as pd
from tqdm import tqdm
import torch
from torch.optim import AdamW, Adam
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, EsmTokenizer

from structevo.reward.prosst.structure.get_sst_seq import SSTPredictor
from structevo.reward.prosst_configuration import ProSSTConfig
from structevo.reward.prosst_modeling import ProSSTForSequenceClassification
from structevo.reward.ggscnn_modeling import GGSCNNPredictor
from structevo.reward.utils import seq_to_one_hot
from structevo.utils import get_logger
log = get_logger("Train Reward")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, 
                        choices=["GB1", "PhoQ", "AAV-medium", "AAV-hard", "GFP-medium", "GFP-hard"])
    parser.add_argument('--round_id', type=int, required=True, default=0)
    parser.add_argument('--seed', type=int, default=42)
    # path args
    parser.add_argument('--train_datafile', type=str, default=None)
    parser.add_argument('--ckpt_load_path', type=str, default=None)
    parser.add_argument('--ckpt_save_path', type=str, default=None)
    parser.add_argument('--structure_datafile', type=str, default=None)
    parser.add_argument('--predict_space_path', type=str, default=None)
    # training args
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--max_protein_len', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=2e-3)
    parser.add_argument('--device', type=str, default='cuda:0')
    return parser.parse_args()


class RewardDataset(Dataset):
    def __init__(self, datafile:str, reward_key="GroundTruth", test_mode=False):
        self.test_mode = test_mode
        df = pd.read_csv(datafile)
        self.seqs = df['Sequence'].tolist()
        if not self.test_mode:
            self.labels = df[reward_key].tolist()
    
    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx):
        x = self.seqs[idx]
        if self.test_mode:
            return x
        else:
            y = self.labels[idx]
            return x, y


def get_dataloader(args):
    log.info("Preparing dataloader...")
    dataset = RewardDataset(args.train_datafile)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    log.info(f"Train Dataset length: {len(dataset)}")
    return dataloader


def get_structure_input_ids(struc_file):
    log.info(f"Loading structure input IDs from {struc_file}")
    if not os.path.exists(struc_file):
        raise FileExistsError("complex_structure.pdb file does not exist:", struc_file)
    predictor = SSTPredictor(structure_vocab_size=2048)
    struc = predictor.predict_from_pdb(struc_file)
    structure_sequence = struc[0]['2048_sst_seq']
    structure_sequence_offset = [i + 3 for i in structure_sequence]
    structure_input_ids = torch.tensor([1, *structure_sequence_offset, 2], dtype=torch.long).unsqueeze(0)
    return structure_input_ids


def get_model_tokenizer_and_optimizer(args):
    args.backbone = "ggscnn" if 'GFP' in args.task or 'AAV' in args.task else "prosst"
    if args.backbone == "ggscnn" and args.round_id == 0:
        log.info("GGSCNN round 0: Copy checkpoint without training")
        ggs_ckpt_path = f"./ckpts/ggs/{args.task}.ckpt"
        shutil.copyfile(ggs_ckpt_path, args.ckpt_save_path)
        exit(0)
    
    if args.backbone == "ggscnn":       # load ckpt
        model = GGSCNNPredictor(args.ckpt_load_path)
        optimizer = Adam(
            model.parameters(), 
            lr=1e-4,
        )
        tokenizer = None
    else:     # prosst: always train from scratch
        assert args.ckpt_save_path.endswith('.csv'), "For ProSST, ckpt_save_path should be a .csv file saving prediction results"
        backbone_ckpt_path = "./ckpts/ProSST-2048"
        config = ProSSTConfig.from_pretrained(backbone_ckpt_path)
        config.num_labels = 1
        model = ProSSTForSequenceClassification.from_pretrained(
            backbone_ckpt_path, 
            config=config
        )
        for params in model.prosst.parameters():
            params.requires_grad = False
        tokenizer = AutoTokenizer.from_pretrained(backbone_ckpt_path)
        tokenizer.model_max_length = args.max_protein_len
        optimizer = AdamW(
            model.parameters(), 
            lr=args.learning_rate, 
            weight_decay=0.001, 
            betas=(0.9, 0.98)
        )

    log.info(model.print_trainable_parameters())
    return model, tokenizer, optimizer


def train(model, dataloader, tokenizer, optimizer, args):
    if args.backbone == "prosst":
        structure_input_ids = get_structure_input_ids(args.structure_datafile)
        structure_input_ids = structure_input_ids.to(args.device)
        structure_len = structure_input_ids.shape[1] - 2
    
    model = model.to(args.device)
    model.train()

    log.info(f"Round {args.round_id} starting training!")
    for epoch in tqdm(range(args.num_epochs)):
        for i, data in enumerate(dataloader):
            seqs, labels = data
            labels = torch.tensor(labels, dtype=torch.float).to(args.device)
            if args.backbone == "prosst":
                if len(seqs[0]) > structure_len:    # For GFP: truncated seqs because tail structure is unavailable
                    seqs = [seq[:structure_len] for seq in seqs]
                tokenized_seqs = tokenizer(seqs, return_tensors='pt').to(args.device)
                outputs = model(
                    input_ids=tokenized_seqs['input_ids'],
                    attention_mask=tokenized_seqs['attention_mask'],
                    ss_input_ids=structure_input_ids,
                    labels=labels  # Add labels for regression loss
                )
                loss = outputs.loss
            else:   # ggscnn
                tokenized_seqs = [seq_to_one_hot(seq) for seq in seqs]
                tokenized_seqs = torch.stack(tokenized_seqs).to(args.device)
                loss = model(tokenized_seqs, labels)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    if args.backbone == "ggscnn":    # save proxy checkpoints
        if not os.path.exists(os.path.dirname(args.ckpt_save_path)):
            log.info(f"Create directory for ckpt: {os.path.dirname(args.ckpt_save_path)}")
            os.makedirs(os.path.dirname(args.ckpt_save_path))
        torch.save(model.state_dict(), args.ckpt_save_path)
        log.info(f"Saved proxy model checkpoint at {args.ckpt_save_path}")
        exit(0)
    else:   # further predict entire space
        return model, structure_input_ids


def inference_entire_space(
            model:ProSSTForSequenceClassification, 
            tokenizer:AutoTokenizer, 
            structure_input_ids:torch.Tensor,
            args
        ) -> float:
    """
    only for prosst in GB1/PhoQ. run inference on all 160,000 seqs and save .csv file
    """
    log.info(f"Loading all sequences from space path: {args.predict_space_path}")
    df_all = pd.read_csv(args.predict_space_path).drop(columns=['Unnamed: 0'])
    test_dataset = RewardDataset(args.predict_space_path, test_mode=True)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    log.info(f"Test Dataset length: {len(test_dataset)}")
    
    predrewards = []
    model.eval()
    log.info(f"Starting prediction for round {args.round_id}")
    for seqs in tqdm(test_dataloader, total=len(test_dataloader)):
        tokenized_seqs = tokenizer(seqs, return_tensors='pt', padding=True, truncation=True).to(args.device)
        with torch.no_grad():
            outputs = model(
                input_ids=tokenized_seqs['input_ids'],
                attention_mask=tokenized_seqs['attention_mask'],
                ss_input_ids=structure_input_ids,
                labels=None
            )
            logits = outputs.logits.squeeze().cpu().detach()
            reward = logits.numpy().tolist()
        predrewards.extend(reward)
    
    df_all['PredictedFitness'] = predrewards
    df_all = df_all.drop(columns=['Sequence'])
    df_all = df_all.sort_values(by='PredictedFitness', ascending=False).reset_index(drop=True)
    df_all.to_csv(args.ckpt_save_path)
    log.info(f"Prediction saved at: {args.ckpt_save_path}")


def main():
    args = get_args()
    print(args)

    dataloader = get_dataloader(args)
    model, tokenizer, optimizer = get_model_tokenizer_and_optimizer(args)
    model, structure_ids = train(model, dataloader, tokenizer, optimizer, args)
    # inference
    if args.backbone == "prosst":
        inference_entire_space(model, tokenizer, structure_ids, args)


if __name__ == "__main__":
    main()
    