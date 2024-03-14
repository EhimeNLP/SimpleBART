import sys
import random
from bert_score import score
import MeCab
from transformers import BertJapaneseTokenizer, BertForMaskedLM
import torch

args = sys.argv

# MeCab
tokenizer = MeCab.Tagger("-d /home/horiguchi/anaconda3/lib/python3.9/site-packages/ipadic/dicdir")
# BERT
model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
bert_tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
bert_mlm = BertForMaskedLM.from_pretrained(model_name)
bert_mlm = bert_mlm.cuda()

# 手法１：単語難易度辞書を使って動的にマスクする
def mask_simp_word(simp):
    surface_simp, base_simp = tokenize(simp)
    count = 0
    for base in base_simp:
        if base in word2complexity:
            if word2complexity[base] == "中級" or word2complexity[base] == "初級":
                count = count + 1
    if count == 0:
        return None
    
    mask_proba = (len(base_simp)/count) * 0.15

    for idx, base in enumerate(base_simp):
        if base in word2complexity:
            if word2complexity[base] == "中級":
                if random.random() < mask_proba * 0.75:
                    surface_simp[idx] = "<MASK>"
            elif word2complexity[base] == "初級":
                if random.random() < mask_proba:
                    surface_simp[idx] = "<MASK>"
    if "<MASK>" in surface_simp:
        return "".join(surface_simp)
    else:
        return None


# 手法２：平易な単語言い換え辞書を使って複雑な単語を平易な単語に置き換えて、マスクする
def replace_comp_word(comp):
    surface_comp, base_comp = tokenize(comp)
    count = 0
    for base in base_comp:
        if base in word2simple:
            count = count + 1
    if count == 0:
        return None, None
    
    mask_proba = (len(base_comp)/count) * 0.15

    replaced_comp = surface_comp.copy()
    masked_comp = surface_comp.copy()
    for idx, base in enumerate(base_comp):
        if base in word2simple:
            tmp = surface_comp.copy()
            tmp[idx] = "[MASK]"
            mask_word = mask_predict("".join(tmp), word2simple[base])
            if mask_word != None and random.random() < mask_proba:
                    replaced_comp[idx] = mask_word
                    masked_comp[idx] = "<MASK>"
    if "<MASK>" in masked_comp:
        return "".join(replaced_comp), "".join(masked_comp)
    else:
        return None, None
    # if calc_bert_score("".join(surface_comp), "".join(replaced_comp)) >= 0.9 and "[MASK]" in masked_comp:
    #     return "".join(replaced_comp), "".join(masked_comp)
    # else:
    #     return None, None


# BERTで穴埋め確率を計算して、言い換え先を決定する
def mask_predict(text, words, tokenizer=bert_tokenizer, bert_mlm=bert_mlm):
    input_ids = tokenizer.encode(text, return_tensors='pt')
    input_ids = input_ids.cuda()
    with torch.no_grad():
        output = bert_mlm(input_ids=input_ids)
        scores = output.logits
    # [MASK]の位置を調べる
    mask_position = input_ids[0].tolist().index(4)
    #スコア順位が一番高いトークンを選択する
    top_word_id = tokenizer.convert_tokens_to_ids(words[0])
    top_word_score = scores[0, mask_position, top_word_id]
    for word in words:
        word_id = tokenizer.convert_tokens_to_ids(word)
        word_score = scores[0, mask_position, word_id]
        if word_score > top_word_score:
            top_word_id = word_id
            top_word_score = word_score
    if tokenizer.convert_ids_to_tokens(top_word_id) == "[UNK]": # BERTの語彙に含まれない場合、[UNK]になるっぽい
        return None
    elif top_word_score < 8:
        return None
    else:
        return tokenizer.convert_ids_to_tokens(top_word_id)
    

def tokenize(text):
    text = tokenizer.parse(text.strip())
    text = text.split("\n")
    surface = []
    base = []
    for line in text[:-2]:
        seg = line.split("\t")
        pos = seg[1].split(",")
        surface.append(seg[0])
        base.append(pos[6])
    return surface, base


# 原文と置き換え後の文の類似度を測定
# def calc_bert_score(comp, replaced_comp):
#     P, R, F1 = score([replaced_comp], [comp], lang="ja")
#     print(comp, replaced_comp)
#     print(F1.item())
#     return F1.item()


# マスク作業
def mask_data(fname1, fname2):
    with open(fname1, "r") as fin1, open(fname2, "r") as fin2, open("../mask_corpus/matcha.dev.8.csv", "w") as fout:
        fout.write("text,label\n")
        for comp, simp in zip(fin1, fin2):
            masked_simp = mask_simp_word(simp)
            replaced_comp, masked_comp = replace_comp_word(comp)
            if masked_simp != None:
                masked_simp = masked_simp.replace('"', "'")
                simp = simp.strip().replace('"', "'")
                fout.write('"' + masked_simp + '","' + simp + '"\n')
            if masked_comp != None:
                masked_comp = masked_comp.replace('"', "'")
                replaced_comp = replaced_comp.replace('"', "'")
                fout.write('"' + masked_comp + '","' + replaced_comp + '"\n')


if __name__ == '__main__':
    # 単語難易度辞書の読み込み
    word2complexity = {}
    with open("../mask_corpus/word2complexity.tsv", "r") as f:
        for line in f:
            line = line.strip().split("\t")
            word2complexity[line[0]] = line[1]   

    # 平易な単語言い換え辞書の読み込み
    threshold = 0.25
    word2simple = {}
    with open("../mask_corpus/word2simple.tsv", "r") as f:
        f = f.read().splitlines()
        for line in f:
            line = line.split("\t")
            if float(line[3]) > threshold:
                word2simple.setdefault(line[0], []).append(line[1])

    mask_data(args[1], args[2])
