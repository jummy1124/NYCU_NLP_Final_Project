from posixpath import split
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, RobertaTokenizer
from torch.utils.data import Dataset
from IPython.display import clear_output
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertForSequenceClassification,BertConfig,RobertaForSequenceClassification
from IPython.display import clear_output
import numpy as np
from sklearn.metrics import accuracy_score,f1_score
from tqdm.notebook import tqdm

tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

# 讀取資料集
test = pd.read_csv("./new_test_0608.csv").drop(columns=['conv_id'])
print("測試樣本數：", len(test))


class FakeNewsDataset(Dataset):
    # 讀取前處理後的 tsv 檔並初始化一些參數
    def __init__(self, mode, df, tokenizer):
        assert mode in ["train", "valid", "test"]
        self.mode = mode
        self.df = df.fillna("")
        self.len = len(self.df)
        self.tokenizer = tokenizer
    
    def __getitem__(self, idx):
        if self.mode == "test":
            text_a, text_b = self.df.iloc[idx, :2].values
            label_tensor = None
        else:
            text_a, text_b, label, segment = self.df.iloc[idx, :].values
            label_tensor = torch.tensor(label)
            
        word_pieces = ["[<s>]"]
        tokens_a = self.tokenizer.tokenize(text_a)
        word_pieces += tokens_a + ["[</s>]"]
        len_a = len(word_pieces)

        tokens_b = self.tokenizer.tokenize(text_b)
        word_pieces += tokens_b + ["[</s>]"]
        len_b = len(word_pieces) - len_a

        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
        segments_tensor = torch.tensor([0] * len_a + [1] * len_b, dtype=torch.long)


        return (tokens_tensor, segments_tensor, label_tensor)
    
    def __len__(self):
        return self.len


def create_mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]
    
    if samples[0][2] is not None:
        # print(samples)
        label_ids = torch.stack([s[2] for s in samples])
    else:
        label_ids = None
    
    tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, batch_first=True)
    if tokens_tensors.size()[1]>512:
        tokens_tensors = tokens_tensors[:,:512]
        segments_tensors = segments_tensors[:,:512]
        # print(tokens_tensors.size())
    
    masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)

    
    return tokens_tensors, segments_tensors, masks_tensors, label_ids


def get_predictions(model, dataloader, compute_acc=False):
    predictions = None
    correct = 0
    total = 0
    running_loss = 0
    ans=[]
    pre=[]
    model.eval()
    with torch.no_grad():
        for data in tqdm(dataloader):
            if next(model.parameters()).is_cuda:
                data = [t.to(device) for t in data if t is not None]
            
            if not compute_acc:
                tokens_tensors, segments_tensors, masks_tensors = data[:3]
                outputs = model(input_ids=tokens_tensors, 
                                # token_type_ids=segments_tensors, 
                                attention_mask=masks_tensors)
                logits = outputs[0]
                _, pred = torch.max(logits.data, 1)
            else:
                tokens_tensors, segments_tensors, masks_tensors, labels = data[:4]
                outputs = model(input_ids=tokens_tensors, 
                                token_type_ids=segments_tensors, 
                                attention_mask=masks_tensors,
                                labels=labels)
                loss = outputs[0]
                logits = outputs[1]
                _, pred = torch.max(logits.data, 1)
                running_loss += loss.item()
                total += labels.size(0)
                correct += (pred == labels).sum().item()
                # print(len(labels.tolist()),len(pred.tolist()))
                for i in range(len(labels.tolist())):
                  ans.append(labels.tolist()[i])
                  pre.append(pred.tolist()[i])
                
            if predictions is None:
                predictions = pred
            else:
                predictions = torch.cat((predictions, pred))
    
    if compute_acc:
        loss = running_loss / total
        valid_f1=f1_score(ans, pre, average='macro')
        return predictions, valid_f1, loss
    
    return predictions


NUM_LABELS = 32
hidden_dropout_prob=0.3

model = RobertaForSequenceClassification.from_pretrained("roberta-large",num_labels=NUM_LABELS,hidden_dropout_prob=hidden_dropout_prob)
cnt=0
for param in model.parameters():
    cnt+=1
    if cnt < 300 : 
        param.requires_grad = False
# model.classifier.out_features=NUM_LABELS
model.load_state_dict(torch.load("./checkpoint_6_0.6529.pth",map_location='cuda:0'))
print(model)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)
model = model.to(device)

model.eval()
clear_output()


testset = FakeNewsDataset("test", test, tokenizer=tokenizer)
testloader = DataLoader(testset, batch_size=32, collate_fn=create_mini_batch)

predictions = get_predictions(model, testloader)
predictions = predictions.cpu().numpy()

sub = pd.read_csv("./fixed_test.csv").drop(columns=['utterance_idx','prompt','utterance'])
sub_df = pd.read_csv("./new_test_0608.csv").drop(columns=['prompt','conv'])
sub_df['pred']=predictions
pd.merge(sub, sub_df).drop(columns=['conv_id']).to_csv('./sub.csv')