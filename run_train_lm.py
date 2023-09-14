import transformers
from transformers.models.gpt2 import GPT2LMHeadModel, GPT2Config
import torch
import argparse
import datetime
import time
import os
import random
import logging
import glob
import json
import numpy as np
from share import get_price

logging.getLogger().setLevel(logging.INFO)

np.random.seed(42)
emb_klts = np.random.random((10, 3))

klt_embs = { 1:emb_klts[0], 
            5:emb_klts[1], 
            15:emb_klts[2], 
            30:emb_klts[3], 
            60:emb_klts[4], 
            101:emb_klts[5], 
            102:emb_klts[6], 
            103:emb_klts[7],
            201:emb_klts[8],  #股票
            202:emb_klts[9]}  #基金

def get_examples(dirname='./share/', bs=4, emb_dim=512):
    
    klt_seqlen = {1:120, 5:120, 15:120, 30:120, 60:120, 101:60, 102:20}
    date_end = datetime.datetime.strptime('2023-08-30 00:00:00', "%Y-%m-%d %H:%M:%S")
    do_rand = False
    input_embs = []
    input_labels = []
    names = glob.glob(dirname+'**/*.txt', recursive=True)
    for i, filestock in enumerate(names):
        _s = filestock.find('share') + 6
        _e = filestock.rfind('/')
        klt = int(filestock[_s:_e])
        if klt in [1, 102]:
            continue
        js_lines = []
        for ii, line in enumerate(open(filestock)):
            js = json.loads(line)
            ##if js['time'] > date_end continue;
            js_lines.append(js)
        if len(js_lines) <= klt_seqlen[klt]:
            continue

        seq_embs = []
        seq_labels = []
        for j in range(1, len(js_lines)):
            if j % 1000 == 0:
                logging.info(f"file {filestock}: line {j}")
            prev_chengjiaoliang = float(js_lines[j-1]['volume'])
            curr_chengjiaoliang = float(js_lines[j]['volume'])
            chengjiaoliang = curr_chengjiaoliang / (prev_chengjiaoliang + curr_chengjiaoliang + 1)
            prev_close = float(js_lines[j-1]['close'])
            if prev_close == 0:
                prev_close += 0.01
            curr_open = float(js_lines[j]['open'])
            curr_high = float(js_lines[j]['high'])
            curr_low  = float(js_lines[j]['low'])
            curr_close = float(js_lines[j]['close'])
            st_open = (curr_open - prev_close ) / prev_close * 100
            st_high = (curr_high - prev_close ) / prev_close * 100
            st_low  = (curr_low - prev_close )  / prev_close * 100
            st_close = (curr_close - prev_close) / prev_close * 100
            label = [st_open, st_high, st_low, st_close]
            
            emb = []
            emb.extend(klt_embs[klt])
            emb.extend(klt_embs[201])
            emb.append(chengjiaoliang)
            emb.extend(label)
            ##others... continue
            emb.extend([0] * (emb_dim - len(emb) ))
            y = label
            seq_embs.append(emb)
            seq_labels.append(y)
            
            if len(seq_labels) == klt_seqlen[klt]: #seq len
                input_embs.append(seq_embs)
                input_labels.append(seq_labels)
                seq_embs = []
                seq_labels = []
            
            if len(input_labels) == bs and not do_rand:
                lens = [len(l) for l in input_labels]
                maxlen =  max(lens)
                input_masks = []
                for k in range(bs):
                    input_masks.extend([1.] * len(input_labels[k]))
                    if len(input_labels[k]) < maxlen:
                        pad_len = maxlen - len(input_labels[k])
                        input_labels[k].extend([ 0., 0., 0., 0.] * pad_len)
                        input_embs[k].extend([ [0.] * emb_dim] * pad_len)
                        input_masks.extend([0.] * pad_len)

                input_embs_t = torch.tensor(input_embs, dtype=torch.float)
                input_labels_t = torch.tensor(input_labels, dtype=torch.float)
                input_masks_t  = torch.tensor(input_masks, dtype=torch.float)
                input_labels = []
                input_embs = []
                yield {"input_emb":input_embs_t, "label":input_labels_t, 'mask':  input_masks_t}

    if do_rand:
        ids = [ i for i in range(0, len(input_labels))]
        random.shuffle(ids)
        input_embs = [input_embs[i] for i in ids]
        input_labels = [input_labels[i] for i in ids]

        for j in range(0, len(ids), bs):
            input_emb = input_embs[j:j+bs]
            input_label = input_labels[j:j+bs]
            
            lens = [len(l) for l in input_label]
            maxlen =  max(lens)
            input_embs_t = torch.zeros(bs, maxlen).float()
            input_labels_t = torch.zeros(bs, maxlen, len(input_label[0][0])).float()
            input_masks  = torch.zeros(bs, maxlen).float()
            for i, _label in enumerate(input_label):
                end = lens[i]
                input_embs_t[i, : end] = torch.tensor(input_emb[i], dtype=torch.float)
                input_labels_t[i, :end] = torch.tensor(input_label[i], dtype=torch.float)
                

            yield {"input_emb":input_embs_t, "label":input_labels_t, 'mask':  torch.ne(input_labels_t, 0).float()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default="cpu", type=str, required=False, help='')
    parser.add_argument('--epochs', default=10, type=int, required=False, help='')
    parser.add_argument('--bs', default=4, type=int, required=False, help='batch size')
    parser.add_argument('--lr', default=1e-3, type=float, required=False, help='')
    parser.add_argument('--warmup_steps', default=3000, type=int, required=False, help='')
    parser.add_argument('--log_step', default=2, type=int, required=False, help='')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--output_dir', default='tmp/', type=str, required=False, help='')
    parser.add_argument('--checkpoint', default='', type=str, required=False, help='checkpoint')
    parser.add_argument('--save_steps', default=1e3, type=int, help='')
    parser.add_argument('--b', default=0.15, type=float, help='')
    parser.add_argument('--seed', default=42, type=int, required=False, help='')
    parser.add_argument('--data', default='./share/', type=str, required=False, help='')
    args = parser.parse_args()

    logging.info('args:\n' + args.__repr__())

    model_conf = {
        "task":"regression",
        "initializer_range": 0.01,
        "layer_norm_epsilon": 1e-05,
        "n_embd": 512,
        "hidden_size": 512,
        "n_head": 4,
        "n_layer": 12,
        "n_positions": 2048,
        "vocab_size": 4,
        "note": "stock prediction using gpt model"
        }
    assert model_conf['n_embd'] == model_conf['hidden_size']

    model_config = GPT2Config.from_dict(model_conf)
    gpu = args.device
    epochs = args.epochs
    bs = args.bs
    lr = args.lr
    warmup_steps = args.warmup_steps
    log_step = args.log_step
    gradient_accumulation = args.gradient_accumulation
    max_grad_norm = args.max_grad_norm
    output_dir = args.output_dir
    b = args.b
    data = args.data
    if  args.checkpoint == "":
        model = GPT2LMHeadModel(config=model_config)
    else:
        logging.info(f'load pretrained model {args.checkpoint}')
        model = torch.load(args.checkpoint, map_location='cpu')
    
    model.train()
    model.to(gpu)
    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    logging.info('number of gpt2 parameters: {}'.format(num_parameters))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, amsgrad=True) 
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=1e8)
    
    logging.info('starting training')
    running_loss = 0
    total_steps  = 0
    step = -1
    
    for epoch in range(epochs):
        logging.info('start epoch {} '.format(epoch + 1))
        t0 = time.time()
        batches = get_examples(dirname=data, bs=bs, emb_dim=model_conf['n_embd'])
        for b, batch in enumerate(batches):
            total_steps += 1
            step += 1
            batch_inputs = batch['input_emb'].to(gpu)
            batch_labels = batch['label'].to(gpu)
            batch_masks  = batch['mask'].to(gpu)
            
            outputs = model.forward(inputs_embeds=batch_inputs, labels=batch_labels, attention_mask=batch_masks)
            loss, logits = outputs[:2]
            if gradient_accumulation > 1:
                loss = loss / gradient_accumulation
            #  loss backward
            
            #loss = torch.abs((loss - b)) + b
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            #  optimizer step
            if (step + 1) % gradient_accumulation == 0:
                running_loss += loss.item()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            if (step+1) % args.save_steps == 0:
                tarp = os.path.join(output_dir, f'model_step_{step+1}.pt')
                logging.info(f'saving model for epoch {epoch+1} step {step+1} loss {running_loss/log_step:.4} path:{tarp}')
                torch.save(model, tarp)
            
            if (step + 1) % log_step == 0:
                logging.info(f'Step {(step + 1) // gradient_accumulation} of total steps {total_steps} of epoch {epoch}, loss {running_loss/log_step:.4}')
                running_loss = 0

        logging.info(f'time for one epoch: {(time.time() - t0):.4} s')

    print('training finished')

def eval():
    pass

#计算loss时按逻辑回归计算，修改modeling_gpt2.py中forward loss计算代码
# loss_fct = MSELoss()
# loss = loss_fct(shift_logits.view(-1), shift_labels.view(-1))

#todo list 
# 1.数据特征及数据api优化完善;
# 2.eval测试;
# 3.gpt2改为RetNet(轻快好) encoder https://github.com/microsoft/torchscale
# 4.增加新闻文本特征，即融入舆情特征(llm+任务微调) 需要gpu算力支持

if __name__ == '__main__':
    main()
