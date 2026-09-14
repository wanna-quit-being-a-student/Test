# CIS — Comprehensive Investment System

Dashboard วิเคราะห์หุ้นกลุ่มเทคโนโลยีใน SET100 (8 หุ้น: ADVANC, CCET, DELTA, HANA, JMART, KCE, THCOM, TRUE)
เพื่อสนับสนุนการตัดสินใจลงทุน สร้างด้วย Streamlit — ข้อมูลทั้งหมดในหน้าจอคำนวณจากไฟล์ Dataset จริง **ไม่มีค่าจำลอง/สุ่มหลงเหลืออยู่**

## โครงสร้างไฟล์

```
cis-dashboard-main/
├── .streamlit/
│   └── config.toml          # ธีม dark navy
├── Dataset/
│   ├── master_all_8_stocks_financials.csv
│   ├── stock_cleaned_data_2016_2025.csv
│   ├── stock_risk_metrics.csv
│   └── train_test/
│       ├── financials_train_2023_2024.csv
│       └── financials_test_2025.csv
├── import_data.py           # นำเข้า CSV -> SQLite (cis_database.db)
├── calculate_scores.py      # คำนวณคะแนน 5 โมดูล + เทรน AI model จริง -> บันทึกผลลัพธ์ลง DB
├── app.py                   # ตัว Dashboard (Streamlit)
├── cis_database.db          # ฐานข้อมูลที่ประมวลผลไว้แล้ว (พร้อมใช้งานทันที)
└── requirements.txt
```

## วิธีรันในเครื่อง (Local)

```bash
pip install -r requirements.txt

# ถ้าต้องการประมวลผลข้อมูลใหม่ทั้งหมด (ไม่บังคับ เพราะ cis_database.db แนบมาให้แล้ว)
python import_data.py
python calculate_scores.py

streamlit run app.py
```

เปิดเบราว์เซอร์ที่ `http://localhost:8501`

## วิธี Deploy บน Streamlit Community Cloud

1. อัปโหลดทั้งโฟลเดอร์นี้ขึ้น GitHub repository (รวมไฟล์ `cis_database.db` เพื่อให้แอปพร้อมใช้ทันทีโดยไม่ต้องรัน pipeline ตอน deploy)
2. ไปที่ [share.streamlit.io](https://share.streamlit.io) → New app → เลือก repo นี้ → Main file path: `app.py`
3. กด Deploy

ถ้าต้องการให้ระบบประมวลผลข้อมูลใหม่ทุกครั้งที่ deploy แทนการแนบ `cis_database.db` ไปด้วย ให้เพิ่มคำสั่งใน `app.py` หรือใช้ GitHub Actions ให้รัน `import_data.py` และ `calculate_scores.py` ก่อน build — แต่โดยทั่วไปแนบไฟล์ `.db` ที่ประมวลผลไว้แล้วไปเลยจะเสถียรและเร็วกว่า

## หลักการคำนวณ (สรุป) — ทุกค่าอิงจากข้อมูลจริงในไฟล์ Dataset

| โมดูล | แหล่งข้อมูล | วิธีคำนวณ (ย่อ) |
|---|---|---|
| 1. Company Health | `stock_financials` (ROE, ROA, D/E, Current Ratio) | ถ่วงน้ำหนัก ROE 30% + ROA 25% + Liquidity 20% + Debt 25% |
| 2. Fair Value | `stock_financials` (FCF, EPS, Total Equity/Liabilities) | Blended DCF (WACC 8.2%, g 2%) 55% + P/E Relative 45% |
| 3. Entry Timing | `stock_daily_prices` (RSI14, MACD, ADX, EMA20/50) | ถ่วงน้ำหนัก RSI 35% + MACD 30% + EMA Trend 35% |
| 4. AI Prediction | `stock_daily_prices` (technical indicators) | Random Forest (train 2023-2024 / test 2025) ทำนายทิศทางราคา 10 วันข้างหน้า |
| 5. Risk Analysis | `stock_risk_metrics` (Beta, Volatility, Max Drawdown) + คำนวณ VaR/Sharpe/Sortino จากราคาจริง | ถ่วงน้ำหนัก Volatility 45% + Max Drawdown 35% + VaR 20% |
| 6. Industry Benchmark | `cis_summary_scores` ของหุ้นทั้ง 8 ตัว | จัดอันดับ (rank) ภายใน sector เดียวกันจากคะแนนจริงทุกโมดูล |

โมเดล AI (Random Forest) และตัวเลข Feature Importance / Accuracy / Precision / Recall / ROC-AUC ที่แสดงในหน้า "AI Prediction"
ล้วนมาจากการเทรนโมเดลจริงในขั้นตอน `calculate_scores.py` (ไม่ใช่ตัวเลขคงที่)

## หมายเหตุสำคัญที่แก้ไขจากเวอร์ชันก่อนหน้า

1. **แก้บั๊ก**: `app.py` เดิมอ้างอิงตาราง `stock_daily` แต่ `import_data.py` สร้างตารางชื่อ `stock_daily_prices` ทำให้แอป error ตอนเปิด — แก้ไขให้ตรงกันแล้ว
2. **เอา `yfinance` ออก**: ตามที่ตกลงกัน ราคาทั้งหมดอ้างอิงจาก Dataset (CSV) เป็นหลัก ไม่พึ่งพา internet ตอน deploy เพื่อความเสถียร
3. **ลบค่าจำลอง/สุ่มทั้งหมด**: Market Cap, P/E, P/B, Revenue Growth, Net Profit Growth, FCF, D/E, Industry Rank, กราฟแท่งเทียน, RSI/MACD/ADX, Key Levels (Support/Resistance), Signal History, Beta/Volatility/Drawdown trend, SHAP-style Feature Importance, Peer Comparison, Radar Chart, Strategic Matrix — **คำนวณจากข้อมูลจริงทั้งหมด**
4. Feature Importance ใช้ `model.feature_importances_` ของ Random Forest จริง (ไม่ใช่ SHAP เนื่องจากไม่ได้ติดตั้งไลบรารี `shap` แต่ให้ผลการตีความลักษณะเดียวกัน)

## การทดสอบ

ไฟล์ `app.py` ผ่านการทดสอบด้วย `streamlit.testing.v1.AppTest` ครบทั้ง 7 หน้า × 8 หุ้น (56 combinations) โดยไม่มี exception เกิดขึ้น
