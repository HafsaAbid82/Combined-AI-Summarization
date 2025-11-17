from huggingface_hub import InferenceClient
import os
from openai import OpenAI
import numpy as np
import torch
from sentence_transformers import util
import csv
import requests
import csv
HF_client = InferenceClient(token=os.environ["HF_TOKEN"])
client = OpenAI()
OpenAI_summaries = []
HF_summaries = []
line_count = 0
fieldnames = ['original_title', 'summary']
datafield = ["title", "author", "date", "description"]
URL = "https://newsapi.org/v2/top-headlines?country=us&apiKey=a1f278e97a06434c8b63036b6830b7c9"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}
response = requests.get(URL, headers=headers)
data = response.json()
results = data["articles"]
title = [articles.get("title") for articles in results]
author = [articles.get("author") for articles in results]
date = [articles.get("publishedAt") for articles in results]
description = [articles.get("description") for articles in results]
with open("news.csv", mode= 'w', newline='', encoding='utf-8') as Writer_file: 
   csv_writer = csv.DictWriter(Writer_file, fieldnames= datafield)
   csv_writer.writeheader()
   data_dicts = [dict(zip(datafield, row)) for row in zip(title, author, date, description)]
   csv_writer.writerows(data_dicts)
   print("Data written to news.csv")
with open("news.csv", mode= 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            line_count += 1
            article_text = (
                    f"title: {row['title']}. "
                    f"author: {row['author']}. "
                    f"date: {row['date']}."
                    f"description: {row['description']}."
                )
            summarization_result = HF_client.summarization(
            article_text, 
            model="facebook/bart-large-cnn"
            )
            summary = summarization_result['summary_text']
            response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a concise summarization assistant. Provide a brief summary."},
                    {"role": "user", "content": f"summarize the following article: {article_text}"}],
                    max_tokens=50,
                )
            OpenAI_summaries.append({
                   "original_title": row['title'],
                    "summary": response.choices[0].message.content
                })
            HF_summaries.append({
                "original_title": row['title'],
                "summary": summary
            })
with open("HF.csv", mode= 'w', newline='', encoding='utf-8') as HF_file: 
   csv_writer = csv.DictWriter(HF_file, fieldnames=fieldnames)
   csv_writer.writeheader() 
   csv_writer.writerows(HF_summaries)
with open("OpenAI.csv", mode= 'w', newline='', encoding='utf-8') as OpenAI_file: 
    csv_writer = csv.DictWriter(OpenAI_file, fieldnames=fieldnames)
    csv_writer.writeheader()
    csv_writer.writerows(OpenAI_summaries) 
hf = [item['summary'] for item in HF_summaries] 
openai = [item['summary'] for item in OpenAI_summaries] 
AI_Summaries = hf + openai
embedding_model = "sentence-transformers/msmarco-bert-base-dot-v5"
BATCH_SIZE = 32
all_embeddings = []
for i in range(0, len(AI_Summaries), BATCH_SIZE):
    batch = AI_Summaries[i:i + BATCH_SIZE]
    try:
        batch_emb = HF_client.feature_extraction(
            batch,
            model=embedding_model, 
        )
        all_embeddings.extend(batch_emb)
        print(f"Successfully processed batch {i // BATCH_SIZE + 1}. Total embeddings: {len(all_embeddings)}")
    except Exception as e:
        print(f"Error processing batch starting at index {i}: {e}")
        break 
query_emb_np = np.array(all_embeddings, dtype=np.float32)
query_emb_tensor = torch.from_numpy(query_emb_np)
N = len(hf)
query = query_emb_tensor[:N] 
doc = query_emb_tensor[N:] 
similarity_matrix = util.cos_sim(query, doc)
AI_similarity_scores = similarity_matrix.diag().cpu().tolist()
print(AI_similarity_scores)





