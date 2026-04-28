import akshare as ak
import pandas as pd
import time
import os
import random
from tqdm import tqdm

#  拦截 VPN 干扰
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""

def fetch_with_resume(start_date="20200101", end_date="20231231"):
    print(" 正在获取沪深300最新成分股列表...")
    try:
        hs300_cons = ak.index_stock_cons(symbol="000300")
        stock_codes = hs300_cons['品种代码'].tolist()
    except Exception as e:
        print(f" 获取名单失败: {e}")
        return

    # 📁 创建一个专门放“碎片”的缓存文件夹
    cache_dir = "data/raw_stocks"
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f" 启动【断点续传】下载模式，共 {len(stock_codes)} 只股票...")

    # 🐜 阶段一：蚂蚁搬家，逐个下载并立即落盘
    for code in tqdm(stock_codes, desc="下载进度"):
        file_path = f"{cache_dir}/{code}.csv"
        
        #  核心防弹逻辑：如果硬盘上已经有这只股票了，直接跳过！
        if os.path.exists(file_path):
            continue
            
        try:
            # 下载数据
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="hfq")
            
            if not df.empty:
                df['ticker'] = code 
                #  立即保存到硬盘，绝不留在内存里过夜！
                df.to_csv(file_path, index=False)
                
            # 随机休眠防封锁 (2~4秒)
            time.sleep(random.uniform(2.0, 4.0)) 
            
        except Exception as e:
            # 如果被踢下线了，打印错误，然后强制冷静 10 秒钟再继续
            print(f"\n 获取 {code} 失败，进入 10 秒冷静期... (错误: {e})")
            time.sleep(10)

    #  阶段二：把所有碎片拼成完整的 3D 面板
    print("\n 所有数据下载完成！开始合并碎片...")
    all_dfs = []
    for file_name in os.listdir(cache_dir):
        if file_name.endswith(".csv"):
            all_dfs.append(pd.read_csv(f"{cache_dir}/{file_name}"))
            
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_path = "data/hs300_panel_raw.csv"
        final_df.to_csv(final_path, index=False)
        print(f" 完美收官！全市场面板数据已合并至: {final_path}")
        print(f" 最终矩阵规模: {final_df.shape}")
    else:
        print(" 缓存文件夹为空，合并失败。")

if __name__ == "__main__":
    fetch_with_resume()