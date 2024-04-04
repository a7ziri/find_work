
import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adagrad
from transformers import AutoTokenizer, AutoModel
from accelerate import Accelerator
from tqdm import tqdm
import random
import numpy as np
import hydra
from omegaconf import DictConfig

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_acc(embeddings):
    batch_size = embeddings.shape[0] // 2
    with torch.no_grad():
        scores = torch.matmul(embeddings[:batch_size].detach(), embeddings[batch_size:].T).cpu().numpy()
    a1 = (scores.argmax(1) == np.arange(batch_size)).mean()
    a2 = (scores.argmax(0) == np.arange(batch_size)).mean()
    return (a1 + a2) / 2

def cleanup():
    torch.cuda.empty_cache()

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
     accelerator = Accelerator()
     device = accelerator.device
     random.seed(cfg.seed)
     np.random.seed(cfg.seed)
     torch.manual_seed(cfg.seed)
     



     df = pd.read_pickle(cfg.data_path)
     all_pairs = list(df.itertuples(index=False, name=None))

     tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
     model = AutoModel.from_pretrained(cfg.model.name)
     model = model.to(device)
     model.train()
     optimizer = Adagrad(
        [p for p in model.parameters() if p.requires_grad],
          lr=cfg.param.learning_rate
     )
     loss_fn = CrossEntropyLoss()

    # Training loop
     num_steps = cfg.param.epochs * len(df) // cfg.param.batch_size
     tq = tqdm(total=num_steps, desc="Training")
     for step in range(num_steps):
        df = df.sample(frac=1).reset_index(drop=True)
        now = df.drop_duplicates(subset='Name', keep='first')
        now = now.drop_duplicates(subset='Description', keep='first')
        all_pairs = list(now.itertuples(index=False, name=None))
        t, q = [list(p) for p in zip(*random.choices(all_pairs, k=cfg.param.batch_size))]
        prepared_data =accelerator.prepare([t, q])    
        try:
                encoded_question = tokenizer(
                    prepared_data[0] + prepared_data[1],
                    padding=True,
                    truncation=True,
                    return_tensors='pt').to(device)
                model_output = model(**encoded_question)
                embeddings = mean_pooling(model_output, encoded_question['attention_mask'])

                all_scores = torch.matmul(embeddings[:cfg.param.batch_size].detach(),
                                          embeddings[cfg.param.batch_size:].T) - torch.eye(cfg.param.batch_size,
                                                                                         device=device) * cfg.param.margin
                loss = loss_fn(all_scores, torch.arange(cfg.param.batch_size, device=device)) + loss_fn(
                    all_scores.T, torch.arange(cfg.param.batch_size, device=device))

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                

                if step % cfg.checkpoint_steps == 0:
                    torch.save(model.state_dict(), cfg.checkpoint_path)

                tq.set_postfix({"Loss": loss.item()})
                tq.update()

        except RuntimeError:
            optimizer.zero_grad(set_to_none=True)
            cleanup()
            continue

     tq.close()

    # Save final model
     torch.save(model.state_dict(), cfg.output_path)

    # Clean up
     cleanup()

if __name__ == "__main__":
    main()