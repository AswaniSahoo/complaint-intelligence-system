# Raw Data

This directory should contain the raw CFPB Consumer Complaint CSV file.

## Download Instructions

### Option 1: CFPB Website (Recommended)
1. Visit the [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
2. Click **"Download the full dataset"** → select **CSV** format
3. Save the file as `complaints.csv` in this directory

### Option 2: Kaggle
1. Visit [CFPB Complaints on Kaggle](https://www.kaggle.com/datasets/selener/consumer-complaint-database)
2. Download and extract the CSV
3. Rename to `complaints.csv` and place in this directory

### Option 3: Direct API
```bash
curl -o complaints.csv "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
unzip complaints.csv.zip
```

## Expected File

```
data/raw/
└── complaints.csv    # ~2GB, ~4 million rows
```

### Required Columns
The pipeline uses these columns from the raw CSV:
- `Consumer complaint narrative`
- `Product`
- `Issue`
- `Date received`

> **Note:** The raw CSV is gitignored due to its large size (~2GB).
> Only the processed data (15K sample) is committed to the repository.
