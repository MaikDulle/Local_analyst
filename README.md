# 📊 Local Analyst

**Analyse your data like a pro — without sending it anywhere.**

Local Analyst is a free, open-source analytics tool that runs entirely on your own computer. Upload a file, click a button, get insights. No cloud. No subscription. No data ever leaves your machine.

Built for marketers, analysts, and business practitioners who want real statistical analysis — not just pretty charts.

---

## What can it do?

Upload any business data file and instantly get:

| What you want to know | What to use |
|---|---|
| What does my data look like? | **Overview** tab |
| How is revenue trending? | **Revenue** tab |
| Which products are performing? | **Products** tab |
| Who are my best customers? | **Customers** tab |
| Are two variables related? | **Correlations** tab |
| Did my A/B test work? | **A/B Testing** tab |
| How long do customers stick around? | **Cohort Analysis** tab |
| How did my campaigns perform? | **Campaign Tracking** tab |
| Where do users drop off? | **Funnel** tab |
| Which channel drives conversions? | **Attribution** tab |
| Is something weird in my data? | **Anomaly Detection** tab |
| Give me everything at once | **Full Scan** tab |

**Optional:** Get plain-English AI explanations of your results — running locally on your machine, completely private.

---

## What it cannot do

- **Connect to live data** — you always upload a file manually
- **Work with huge datasets** — best up to ~500,000 rows; larger files will be slow
- **Connect to the internet** — it's fully offline by design
- **Replace a data scientist** — it highlights patterns, you decide what they mean
- **Guarantee perfect AI output** — the AI interpretation feature can be wrong; always check the actual numbers
- **Run on a mobile device** — it was developed as a desktop tool
---

## What files can I upload?

| File type | Extension | Notes |
|---|---|---|
| Excel | `.xlsx`, `.xls`, `.xlsm` | Multi-sheet supported |
| CSV / text | `.csv`, `.tsv`, `.txt` | Auto-detects separators |
| JSON | `.json` | Handles nested structures |
| PDF reports | `.pdf` | Extracts tables automatically |
| PowerPoint | `.pptx` | Extracts tables + chart data from slides |
| Word | `.docx` | Extracts tables and text |

> **PDF and PowerPoint tip:** The tool can extract both text-based tables AND image-based charts (using OCR). So even if your slides have screenshots of charts, it will try to read them.

---

## System requirements

| | Minimum |
|---|---|
| Operating system | Windows 10/11, macOS 12+, or Linux |
| Python | **3.10 or 3.11 or higher** |
| RAM | 4 GB (8 GB recommended, 16 GB if using AI) |
| Disk space | ~2 GB for the app + ~1 GB if you add the AI model |
| Internet | Only needed once — to download the tool and packages |

---

# Installation — Getting Local Analyst Running on Your Computer

> This takes about **10–15 minutes** and you only do it **once**.  
> You don't need any technical background. Just follow the steps in order.

---

## Before you start — what you'll need

- A laptop or desktop running **Windows 10/11** or **macOS**
- An internet connection (only needed during setup)
- About **3 GB of free disk space**

---

## Step 1 — Install Python

Python is the engine that powers Local Analyst. Think of it like installing a printer driver — you don't need to understand it, you just need it to be there.

1. Open your browser and go to: **[python.org/downloads](https://www.python.org/downloads/)**
2. Click on 'get the standaline installer for **Python 3.XX.X** - Link
3. Open the file that downloads
4. ⚠️ **Before clicking anything else:** look at the bottom of the installer window and **tick the box that says "Add Python to PATH"**  
   *(If you skip this, the setup won't work)*
5. Click **"Install Now"** and wait for it to finish
6. Click **Close**

---

## Step 2 — Download Local Analyst

1. Go to: **[github.com/MaikDulle/Local_analyst](https://github.com/MaikDulle/Local_analyst)**
2. Click the green **"Code"** button (top right of the file list)
3. Click **"Download ZIP"**
4. Once downloaded, **right-click the ZIP file → Extract All** (Windows) or **double-click** it (Mac)
5. Move the extracted folder somewhere easy to find — for example, your **Desktop** or your **Documents** folder  
   *(The folder is called `Local_analyst-main` — you can rename it to just `Local_analyst` if you like)*

---

## Step 3 — Open a terminal inside the folder

This is the step most people are nervous about. A terminal is just a text box where you give your computer instructions. Here's the easiest way to open one already pointing at the right place:

**On Windows:**
1. Open the `Local_analyst` folder in File Explorer
2. Click in the **address bar** at the top (where it shows the folder path)
3. Type `cmd` and press **Enter**
4. A black window opens — that's the terminal, and it's already in the right folder ✓

**On Mac:**
1. Open the `Local_analyst` folder in Finder
2. Right-click (or Control-click) anywhere inside the folder — **not** on a file
3. Select **"New Terminal at Folder"**  
   *(If you don't see this option: go to System Settings → Privacy & Security → Developer Tools and enable Terminal)*

---

## Step 4 — Run the setup (one command, one time)

Copy the line below, paste it into your terminal window, and press **Enter**:

```
python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt
```

**On Mac, use this version instead:**

```
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```

This will take **3–8 minutes** — you'll see a lot of text scrolling past. That's normal. Wait until it stops and you see a blinking cursor again.

> 💡 **What is this doing?** It's creating a protected space for Local Analyst's packages (so they don't interfere with anything else on your computer), then downloading everything the app needs to run.

---

## Step 5 — Start the app

Once setup is done, type this into the same terminal and press **Enter**:

```
streamlit run app/main.py
```

**On Mac:**
```
streamlit run app/main.py
```

Your browser will open automatically and show the Local Analyst interface. If it doesn't open by itself, go to your browser and type `http://localhost:8501` in the address bar. Repeat step 5 for opening the app next time.

🎉 **You're done.** Upload a file in the left sidebar and start exploring your data.

---

## How to use it

### 1. Upload your file

Use the file uploader in the left sidebar. The tool will automatically read the file and show a preview.

If your file has multiple tables (common in PDFs and PowerPoints), a **table selector** will appear in the sidebar so you can switch between them.

---

### 2. Map your columns (optional)

The tool tries to auto-detect which column is your date, revenue, customer ID, etc. You can correct these in the sidebar if it gets it wrong. This unlocks more specific analyses in each tab.

You don't need to map everything — most tabs will still work with whatever columns are available.

---

### 3. Pick a tab and click Analyze

Each tab has a button to run its analysis. Results appear below it. That's it.

---

## What each tab does

#### 📋 Overview
A quick snapshot of your data: how many rows and columns, what's in each column, missing values, basic statistics. Good first stop after uploading.

#### 💰 Revenue
Revenue trends over time, Pareto analysis (which 20% of products/customers generate 80% of revenue), growth metrics, and period comparisons. Needs a date column and a revenue column mapped.

#### 📦 Products
Top-performing products, category breakdowns, revenue contribution by product. Needs a product or category column mapped.

#### 👥 Customers
RFM segmentation (groups customers into 11 types: Champions, Loyal, At Risk, etc.), customer value tiers, and churn risk scoring. Needs a customer ID and a date column.

#### 🔗 Correlations
Shows which columns in your data are related to each other — and how strongly. Works with all data types: numbers, categories, and mixed. Always shows the full heatmap so you can see even weak relationships.

#### 🧪 A/B Testing
Statistical significance testing for A/B experiments. Tells you if the difference between your control and variant is real or just random noise. Shows lift %, p-value, confidence intervals, and a recommendation.

#### 📊 Cohort Analysis
Groups customers by when they first appeared and tracks their behaviour over time. Great for understanding retention, churn timing, and customer lifetime value.

#### 🎯 Campaign Tracking
Year-over-year performance comparison, wave/season analysis, and campaign KPIs (CTR, CVR, CPC, CPA, ROAS). Works well with campaign or advertising data.

#### 🔀 Funnel
Shows drop-off rates through a sequence of steps (e.g. Visit → Sign Up → Purchase). Identifies where you lose the most users.

#### 📡 Attribution
Compares different attribution models (Last Touch, First Touch, Linear, Time Decay, U-Shape) to understand which channels deserve credit for conversions.

#### 🚨 Anomaly Detection
Finds unusual values, unexpected patterns, and statistical outliers in your data. You can choose sensitivity level (low / medium / high) and anomaly types.

#### 🔎 Full Scan
Runs all checks at once — trends, anomalies, correlations, data quality, outliers, missing data — and presents everything as a prioritised list with severity ratings. Good starting point if you don't know where to look.

---

## Optional: AI interpretation

The AI feature adds a **"Get AI Interpretation"** button to the main tabs. Click it after running an analysis and the AI will write a short plain-English explanation of the results.

**The AI runs entirely on your machine** — no API key, no internet connection, no data sent anywhere.

> ⚠️ The AI can be wrong or incomplete. Always check the actual numbers above any AI explanation. It is a helper, not a source of truth.

### How to set up AI (takes about 10 minutes)

**Step 1 — Install the AI engine**

In your terminal (with the virtual environment active):

```
pip install llama-cpp-python --only-binary=llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

> This installs a pre-built package — no C++ compiler or technical setup needed.

**Step 2 — Download a language model**

```
python download_model.py
```

This shows a menu of models. Press Enter to download the recommended one (Qwen 2.5 · 1.5B, ~1 GB). The model is saved to the `models/` folder automatically.

**Step 3 — Enable AI in the app**

1. Open the app
2. In the left sidebar, scroll down to **AI Settings**
3. Change backend from "Rule-based (no LLM)" to **"Local LLM"**
4. The model you downloaded will appear in the dropdown — select it
5. The "Get AI Interpretation" button now appears in Overview, Correlations, Anomaly Detection, and Full Scan tabs

---

## Tips for best results

- **Column names matter.** The cleaner your column names (e.g. `date`, `revenue`, `customer_id`), the better the auto-detection works. Avoid special characters.
- **Dates should be a real date format.** `2024-01-15` or `15/01/2024` both work. Plain year numbers like `2024` may not be recognised as dates.
- **Start with Full Scan** if you're not sure what to look for. It takes 30 seconds and gives you a prioritised list of findings.
- **If a tab shows nothing**, check that the relevant columns are mapped in the sidebar. For example, Cohort Analysis needs both a customer column and a date column.
- **Large files are slower.** If your file has more than 100,000 rows, expect a few seconds of loading time per analysis.

---

## Troubleshooting

**"Python is not recognized" or "python not found"**
→ Python was not added to PATH during installation. Reinstall Python and make sure to tick "Add Python to PATH" on the first screen.

**"pip is not recognized"**
→ Same issue. Reinstall Python with PATH option checked, then restart your terminal.

**"ModuleNotFoundError" when starting the app**
→ Your virtual environment might not be active. Run `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux) first, then try again.

**App opens but shows a blank screen or error**
→ Try refreshing the browser. If the problem persists, stop the terminal with `Ctrl+C` and restart with `streamlit run app/main.py`.

**PDF or PowerPoint shows "No tables found"**
→ The file may consist entirely of images with no extractable text. Make sure `easyocr` installed correctly (`pip install easyocr`). First-time OCR is slow (the model downloads automatically).

**AI interpretation button doesn't appear**
→ Make sure the backend in AI Settings is set to "Local LLM" (not "Rule-based"). Also confirm that `llama-cpp-python` is installed and a `.gguf` model file is in the `models/` folder.

**AI is very slow**
→ Language models run on your CPU, so the first response takes 20–60 seconds. Subsequent responses in the same session are faster because the model stays loaded in memory.

---

## Project structure (for the curious)

```
Local_analyst/
├── app/                     ← The web interface (main.py)
├── analysis_engine/         ← All statistical analysis logic
├── data_upload_engine/      ← File readers (CSV, Excel, PDF, PPTX, etc.)
├── viz_engine/              ← Charts and visualisations
├── ai/                      ← AI interpretation (optional)
├── models/                  ← Put your .gguf model files here
├── data/                    ← Sample files for testing
├── download_model.py        ← Downloads an AI model automatically
└── requirements.txt         ← List of required packages
```

---

## License

MIT — free to use, modify, and share.

---

## AND...

> **Remember:** AI is a tool for interpretation, not calculation. Trust the numbers, question the narrative.
