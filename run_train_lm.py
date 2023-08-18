import transformers
from transformers.models.gpt2 import GPT2LMHeadModel, GPT2Config
import torch
import argparse
from datetime import datetime
import time
import os
import logging
import efinance as ef

logging.getLogger().setLevel(logging.INFO)

def get_examples(stocks=['601919', ], bs=1, seq=128, emb_dim=512):
    
    klts_emb = { 1:[0, 0, 0], 5:[ 0, 0, 1.], 15:[ 0, 1., 0], 30:[0, 1., 1.], 
                60:[1., 0, 0], 101:[ 1., 0, 1.], 102:[ 1., 1., 0], 103:[1., 1., 1.], }
    klts = [5]   #1分钟 5分钟 15分钟 30分钟 60分钟 1天 1周 1月
    #beg = '20010101'
    #end = '20230809'
    input_embs = []
    input_labels = []
    for i, klt in enumerate(klts):
        for j, stock in enumerate(stocks):
            df = ef.stock.get_quote_history(stock, klt=klt, )  #@指定日期接口不生效
            seq_embs = []
            seq_labels = []
            for k in range(1, df.shape[0]):
                prev_chengjiaoliang = df.loc[k-1]['成交量']
                curr_chengjiaoliang = df.loc[k]['成交量']
                chengjiaoliang = df.loc[k]['成交量'] / (prev_chengjiaoliang+1)
                zhenfu = df.loc[k]['振幅']
                zhangdiefu = df.loc[k]['涨跌幅']
                label = df.loc[k]['涨跌幅']
                
                emb = []
                emb.extend(klts_emb[klt])
                emb.append(chengjiaoliang)
                emb.append(zhenfu)
                emb.append(zhangdiefu)
                ##others... continue
                emb.extend([0] * (emb_dim - len(emb) ))
                y = label
                seq_embs.append(emb)
                seq_labels.append(y)
                
                if len(seq_labels) == seq:
                    input_embs.append(seq_embs)
                    input_labels.append(seq_labels)
                    seq_embs = []
                    seq_labels = []

                    if len(input_embs) == bs:
                        input_embs_t = torch.tensor(input_embs, dtype=torch.float32)
                        input_labels_t = torch.tensor(input_labels, dtype=torch.float32)
                        input_embs = []
                        input_labels = []
                        yield {"input_emb":input_embs_t, "label":input_labels_t }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default="cpu", type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--epochs', default=10, type=int, required=False, help='训练循环')
    parser.add_argument('--bs', default=4, type=int, required=False, help='训练batch size')
    parser.add_argument('--lr', default=1e-3, type=float, required=False, help='学习率')
    parser.add_argument('--warmup_steps', default=3000, type=int, required=False, help='warm up步数')
    parser.add_argument('--log_step', default=2, type=int, required=False, help='')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='梯度积累')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--output_dir', default='tmp/', type=str, required=False, help='模型输出路径')
    parser.add_argument('--checkpoint', default='', type=str, required=False, help='checkpoint')
    parser.add_argument('--save_steps', default=1e3, type=int, help='')
    parser.add_argument('--seed', default=42, type=int, required=False, help='')
    args = parser.parse_args()

    logging.info('args:\n' + args.__repr__())

    model_conf = {
        "initializer_range": 0.01,
        "layer_norm_epsilon": 1e-05,
        "n_embd": 512,
        "hidden_size": 512,
        "n_head": 4,
        "n_layer": 12,
        "n_positions": 2048,
        "vocab_size": 1,
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr) 
    #torch.optim.Adam(model.parameters(), lr=lr, eps=1e-8, amsgrad=True) 
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=1e8)
    
    logging.info('starting training')
    running_loss = 0
    total_steps  = 0
    step = -1
    for epoch in range(epochs):
        logging.info('epoch {}'.format(epoch + 1))
        t0 = time.time()
        batches = get_examples(bs=bs, emb_dim=model_conf['n_embd'])
        for b, batch in enumerate(batches):
            total_steps += 1
            step += 1
            
            batch_inputs = batch['input_emb'].to(gpu)
            batch_labels = batch['label'].to(gpu)
            
            outputs = model.forward(inputs_embeds=batch_inputs, labels=batch_labels)
            loss, logits = outputs[:2]
            if gradient_accumulation > 1:
                loss = loss / gradient_accumulation
            #  loss backward
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
                logging.info(f'Input Shape {batch_inputs.shape} Step {(step + 1) // gradient_accumulation} of total steps {total_steps} of epoch {epoch}, loss {running_loss/log_step:.4}')
                running_loss = 0
        logging.info('epoch {} finished'.format(epoch + 1))

        logging.info(f'time for one epoch: {(time.time() - t0):.4} s')

    print('training finished')

#计算loss时按逻辑回归计算，修改modeling_gpt2.py中forward loss计算代码
# loss_fct = MSELoss()
# loss = loss_fct(shift_logits.view(-1), shift_labels.view(-1))

#todo list 
# 1.数据特征及数据api；
# 2.eval测试;
# 3.gpt2改为RetNet(更轻更快效果也更好) encoder https://github.com/microsoft/torchscale
# 4.增加新闻文本特征，即融入舆情特征(llm+任务微调) 需要gpu算力支持

if __name__ == '__main__':
    main()
