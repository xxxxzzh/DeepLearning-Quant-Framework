import baostock as bs
import pandas as pd
import os
from tqdm import tqdm

def fetch_real_market_data(start_date="2020-01-01", end_date="2023-12-31"):
    print(" 启动 Baostock 量化数据专线...")
    
    # 1. 登录系统 (匿名登录，无限制)
    lg = bs.login()
    if lg.error_code != '0':
        print(f"❌ 登录失败: {lg.error_msg}")
        return

    # 2. 获取沪深300成分股名单
    print("⏳ 正在获取沪深300成分股...")
    rs = bs.query_hs300_stocks()
    stock_codes = []
    while (rs.error_code == '0') & rs.next():
        # 获取代码，比如 'sh.600000'
        stock_codes.append(rs.get_row_data()[1]) 
        
    print(f" 成功拿到 {len(stock_codes)} 只真实股票名单！开始全速下载...")

    all_data = []
    
    # 3. 批量下载 (不需要休眠！不需要热点！直接满速跑！)
    for code in tqdm(stock_codes, desc="真实数据下载进度"):
        # adjustflag="1" 表示后复权 (和咱们之前 AkShare 的 hfq 对应)
        rs_data = bs.query_history_k_data_plus(
            code,
            "date,code,open,high,low,close,volume,amount,pctChg",
            start_date=start_date, end_date=end_date,
            frequency="d", adjustflag="1"
        )
        
        data_list = []
        while (rs_data.error_code == '0') & rs_data.next():
            data_list.append(rs_data.get_row_data())
            
        if data_list:
            df = pd.DataFrame(data_list, columns=rs_data.fields)
            # 【核心清洗】Baostock 的代码带前缀(sh.600000)，我们要切掉前3个字符，变成 600000
            df['ticker'] = df['code'].str[3:] 
            all_data.append(df)

    # 4. 合并与保存
    print("\n 所有数据下载完成！正在压制 3D 面板...")
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        
        # 将列名统一修改为我们清洗脚本认识的名字
        final_df.rename(columns={'pctChg': 'pct_chg'}, inplace=True)
        
        # Baostock 下来的数据默认是字符串，我们需要强转为数字
        cols_to_numeric = ['open', 'high', 'low', 'close', 'volume', 'pct_chg']
        for col in cols_to_numeric:
            final_df[col] = pd.to_numeric(final_df[col], errors='coerce')

        # 落地保存，直接覆盖刚才的假数据
        os.makedirs("data", exist_ok=True)
        save_path = "data/hs300_panel_raw.csv"
        final_df.to_csv(save_path, index=False)
        
        print(f" 真实全市场数据获取成功！已保存至: {save_path}")
        print(f" 最终矩阵规模: {final_df.shape}")
    else:
        print(" 未获取到任何数据。")

    # 5. 登出系统
    bs.logout()

if __name__ == "__main__":
    fetch_real_market_data()