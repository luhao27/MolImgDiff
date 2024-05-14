from xmlrpc.client import FastMarshaller
from transformers import BertTokenizer, AutoTokenizer,BertModel, AutoModel 
import os
import json
import time
import random


tokenizer = AutoTokenizer.from_pretrained("esm/esm")
model = AutoModel.from_pretrained("esm/esm")


def get_emb(fasta, length = 512):
    # fasta = "MEILCEDNIS<pad>"
    # 获取fasta长度
    if length > len(fasta):
        pad_len = length - len(fasta)
        fasta = fasta + '<pad>' * pad_len
    
    seed = random.randrange(0, 40)
    
    global model, tokenizer
    outputs = model(**tokenizer(fasta, return_tensors='pt'))
    
    

    outputs.last_hidden_state
   
    embedding = outputs.last_hidden_state.squeeze(0)[1:]
    # print(embedding.shape)
    embedding = embedding[:-1]

    return embedding.data.numpy().tolist()
    

if __name__ == "__main__":
    
    # this is train pdb data
    dict_path = "../datasets/chembel_512/dict.json"
    pdb_file = '../datasets/chembel_512/pdb_new_512_uni.smi'

    emb_path = "../datasets/chembel_512/dict_bedding.json"
    
    
    # this is sarv_cov_2
    # dict_path = "dict_conv.json"
    # pdb_file = 'pdb_conv_512.smi'
    # emb_path = "dict_conv_bedding.json"


    pdb_arr = open(pdb_file, 'r').readlines()
    pdb_arr = list(set(pdb_arr))
    print(len(pdb_arr))
    
 


    dict_bed = {}
    with open(dict_path, 'r') as f:
        dict = json.load(f)
        print(len(dict.keys()))
        
        for idx, pdb in enumerate(pdb_arr):
            fasta = dict[pdb.strip()]
            bedding = get_emb(fasta=fasta)
            dict_bed[pdb.strip()] = bedding
            
            if idx%10==0:
                print(idx)
                now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print(now)


# write json file
json_str = json.dumps(dict_bed)
with open(emb_path, "w") as f:
    f.write(json_str)   
