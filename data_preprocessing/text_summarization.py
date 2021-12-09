from tqdm import tqdm
from summarizer import Summarizer
from transformers import BartTokenizerFast, BartForConditionalGeneration, BartConfig
from transformers import BertModel, BertTokenizerFast, BertConfig
import json
import os


def generate_extractive_summary(text, model_name='bert-base-uncased', random_state=43, ratio=0.4):
    config = BertConfig.from_pretrained(model_name)
    config.output_hidden_states = True
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name, config=config)
    extractive_summary_model = Summarizer(
        custom_model=model,
        custom_tokenizer=tokenizer,
        random_state=random_state
    )
    return [extractive_summary_model(t, ratio=ratio) for t in tqdm(text)]


def generate_abstractive_summary(text, model_name='sshleifer/distilbart-cnn-12-6'):
    sum_tokenizer = BartTokenizerFast.from_pretrained(model_name)
    sum_model = BartForConditionalGeneration.from_pretrained(model_name)
    inputs = [sum_tokenizer(t, return_tensors="pt", truncation=True, padding=True, max_length=1024).input_ids for t in text]
    outputs = [sum_model.generate(i, min_length=int(len(text[idx].split()) * 0.4), max_length=512, top_k=100, top_p=0.95, do_sample=True) for idx, i in enumerate(tqdm(inputs))]
    summarized_texts = [sum_tokenizer.batch_decode(o, skip_special_tokens=True) for o in outputs]
    tmps = [s[0].strip() for s in summarized_texts]
    return tmps


def save_summary(news_id, summary_text, summary_type, dataset='politifact', subset='fake'):
    folder_path = '../data/fakenewsnet_dataset/' + dataset + '/' + subset + '/' + news_id
    if os.path.isdir(folder_path):
        file_path = folder_path + '/summary.json'
        if os.path.isfile(file_path):
            with open(file_path) as json_file:
                data = json.load(json_file)
            data[summary_type] = summary_text
        else:
            data = {summary_type: summary_text}
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    else:
        print(f"[WARNING] folder '{folder_path}' doesn't exist")
