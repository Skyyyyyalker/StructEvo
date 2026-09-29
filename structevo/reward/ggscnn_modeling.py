import torch
from structevo.reward.utils import BaseCNN

class GGSCNNPredictor(torch.nn.Module):
    def __init__(self, predictor_path: str=None):
        super(GGSCNNPredictor, self).__init__()

        self.model = BaseCNN(make_one_hot=False)
        model_ckpt = torch.load(predictor_path, map_location="cpu")
        if "state_dict" in model_ckpt.keys():
            model_ckpt = model_ckpt["state_dict"]
        model_ckpt = {k.replace('predictor.',''):v for k,v in model_ckpt.items()}
        model_ckpt = {k.replace('model.',''):v for k,v in model_ckpt.items()}
        self.model.load_state_dict(model_ckpt)
    
        self.loss_fn = torch.nn.MSELoss()
    
    def forward(self, seqs, labels):
        preds = self.model(seqs)
        loss = self.loss_fn(preds, labels)
        return loss
    
    def print_trainable_parameters(self):
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print_str = (f"trainable params: {trainable_params} || " 
               f"all params: {all_param} || "
               f"trainable%: {100 * trainable_params / all_param}")
        return print_str

    # @torch.no_grad()
    # def predict(self, seqs):
    #     """ get oracle scores for sequences """
    #     tokenized_seqs = self.tokenize(seqs)
    #     batches = torch.split(tokenized_seqs, self.batch_size, 0)
    #     scores = []
    #     for b in tqdm(batches, total=len(batches), desc="Oracle scoring"):
    #         if b is None: continue
    #         results = self.oracle_model(b).detach()
    #         scores.append(results)
    #     return torch.concat(scores, dim=0)