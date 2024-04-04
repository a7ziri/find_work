import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer , AutoModel
import numpy as np
import hydra
from omegaconf import DictConfig

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    data = pd.read_csv(cfg.data.csv_path)
    data_emb = data['0'].tolist()
    model =  SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    embeddings = model.encode(data_emb ,convert_to_tensor=True, normalize_embeddings=True)
    embeddings =embeddings.cpu()
    np.save('emb_for_model_mini_norm.npy' , embeddings)





if __name__ == "__main__":
    main()