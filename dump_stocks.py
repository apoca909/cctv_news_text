import efinance as ef
from share import get_price
import json
import datetime
import os

def get_all_stocks():
    markets = {'深A': "XSHE", "沪A": 'XSHG'}
    stock_infos = ef.stock.get_realtime_quotes()
    x, y = stock_infos.shape
    stock_tuples = []
    for i in range(0, x):
        stock_id = stock_infos.loc[i]['股票代码'] + '.' + markets[
            stock_infos.loc[i]['市场类型']]
        stock_name = stock_infos.loc[i]['股票名称']
        stock_tuples.append((stock_id, stock_name, 0))

    return stock_tuples


def get_all_funds():
    fund_infos = ef.fund.get_fund_codes()
    x, y = fund_infos.shape
    fund_tuples = []
    for i in range(x):
        fund_id = fund_infos.loc[i]['基金代码']
        fund_name = fund_infos.loc[i]['基金简称']
        fund_tuples.append((fund_id, fund_name, 1))
    return fund_tuples


date_1 = datetime.datetime.strptime('2023-08-30 00:00:00', "%Y-%m-%d %H:%M:%S")
date_2 = datetime.datetime.strptime('2023-09-13 15:00:00', "%Y-%m-%d %H:%M:%S")
stocks = get_all_stocks()
klts = [1, 5, 15, 30, 60, 101, 102]
for klt in klts[6:]:
    for st in stocks:
        print(st, date_1, klt)
        stock_id = st[0]
        try:
            if klt < 101:
                date_end = date_2
            else:
                date_end = date_1
            df = get_price(stock_id, date_end, count=800, frequency=klt)
        except Exception as e:
            print("err.", st, date_end, klt)
            continue
        if df.shape[0] == 0:
            print("empty.", st, date_end, klt)
            continue
        if not os.path.exists(f'./share/{klt}'):
            os.mkdir(f'./share/{klt}')
        if os.path.exists(f'./share/{klt}/{stock_id}.txt'):
            all_times = [
                json.loads(line)['time'] for line in open(
                    f'./share/{klt}/{stock_id}.txt', 'r', encoding='utf-8')
            ]
            last_time = all_times[-1] if len(all_times) > 0 else "2000-01-01 00:00:00"
            last_time = datetime.datetime.strptime(last_time, "%Y-%m-%d %H:%M:%S")

        else:
            last_time = datetime.datetime.strptime("2000-01-01 00:00:00",
                                                   "%Y-%m-%d %H:%M:%S")

        fw = open(f'./share/{klt}/{stock_id}.txt', 'a+', encoding='utf-8')

        for k in range(0, df.shape[0]):
            t = df.index[k]
            if t <= last_time:
                continue
            js = {
                'id': st[0],
                'name': st[1],
                'time': str(t),
                'open': df.iloc[k]['open'],
                'close': df.iloc[k]['close'],
                'high': df.iloc[k]['high'],
                'low': df.iloc[k]['low'],
                'volume': df.iloc[k]['volume']
            }
            print(json.dumps(js, ensure_ascii=False), file=fw)
        fw.flush()
