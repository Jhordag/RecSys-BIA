
#!python /content/recsys-challenge-RL/baseline/extract_sentece_transformer.py sentence-transformers/all-MiniLM-L6-v2 '/content/drive/MyDrive/Trabalho do Ricardo/df_categories.csv' output_path

import argparse
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def load_dataset(csv_file: str):
  """
  Load the dataset from a csv file.
  """
  df = pd.read_csv(csv_file)[["business_id", "name", "cate","features"]]
     
  return df

def load_model(model_base: str):
  """
  Load the model from https://github.com/huggingface/transformers

  """
  model = SentenceTransformer(model_base).to(device)
  return model


def get_embeddings(model, df: pd.DataFrame, column: str):

  data = []
  for i, row in tqdm(df.iterrows(), total=df.shape[0]):
   data.append(model.encode(str(row[column])))
  return data

def export_dataset(df: pd.DataFrame, emb_column: str, output_file: str):
  """
  Export the embeddings to a csv file.
  """
  np.savetxt(output_file+'/embeddings.txt', np.stack(df[emb_column]), delimiter='\t')
  df.drop(emb_column, axis=1).to_csv(output_file+"/metadados.csv", sep="\t", index=False)

if __name__ == '__main__':
  """
  Extract the embeddings from a dataset - baseline code.
  
  Params:
  
  model_base: The model base to extract the embeddings.
  csv_file: The csv file to extract the embeddings.
  output_path: The output path to save the embeddings and metadata.
  """

  parser = argparse.ArgumentParser()

  parser.add_argument('model_base',type=str,help='Transform Model Base',default='bert-base-uncased',)
  parser.add_argument('csv_file',type=str,help='The csv file',)
  parser.add_argument('output_path',type=str,help='Output Path',)

  args = parser.parse_args()

  # Load Dataset
  print("\n\nLoad Dataset...")
  df = load_dataset(args.csv_file)
  print(df.head()) 

  # Load Model
  print("\n\nLoad Transform Model...")
  model= load_model(args.model_base)  
  print(model)

  # Extract Embeddings
  print("\n\nExtract Embeddings...")
  df["embs"] = get_embeddings(model, df, "features")
  df["embs"] = df["embs"].apply(np.array)
  print(df.head())

  #Exporta Dataset
  print("\n\nExtract Dataset...")

  export_dataset(df, "embs", args.output_path)

  print("\n\nDone! :)")
