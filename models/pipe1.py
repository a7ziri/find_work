import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer , AutoModel
import numpy as np
import hydra
from omegaconf import DictConfig
@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    data = pd.read_csv(cfg.data.csv_path)
    data_emb = data['0'].tolist()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    model = AutoModel.from_pretrained(cfg.model.name)
    tokenized_texts = tokenizer(
    data_emb, max_length=512, return_tensors="pt", padding=True, truncation=True
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    features = []
    batch_size = 8
    with torch.no_grad():
        for i in range(0, len(data_emb), batch_size):
            texts_batch = tokenized_texts["input_ids"][i : i + batch_size].to(device)
            masks_batch = tokenized_texts["attention_mask"][i : i + batch_size].to(device)
            output = model(texts_batch, masks_batch)
            batch_features = output.last_hidden_state[:, 0, :].cpu().numpy()
            features.append(batch_features)

    features = np.concatenate(features, axis=0)
    np.save('emb_for_model_deep.npy' , features)

if __name__ == "__main__":
    main()