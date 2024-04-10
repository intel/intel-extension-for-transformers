# Extract Tables From PDF File

We leveraged [table-transformer](https://github.com/microsoft/table-transformer) for tables extraction and adapted the multi-page table solution of [Amazon Textract response parser](https://github.com/aws-samples/amazon-textract-response-parser) library to table-transformer for handling multi-page table.

## Prepare Environment

```
pip install -r requirements.txt
```
Note that additional language library of tesseract is needed for handling language other than English, for example, for Simplified Chinese, below library is needed.
```
apt-get install tesseract-ocr-chi-sim
```

## Prepare Models

```
git clone https://huggingface.co/bsmock/tatr-pubtables1m-v1.0
git clone https://huggingface.co/bsmock/TATR-v1.1-All
```

## Usage

### Run the table extraction script
For local pdf file, run below command:
```
python extract_tables.py --pdf_file /path/to/pdf_file --structure_model_path TATR-v1.1-All/TATR-v1.1-All-msft.pth  --detection_model_path tatr-pubtables1m-v1.0/pubtables1m_detection_detr_r18.pth  -c
```

For url of pdf file, run below command:
```
python extract_tables.py --pdf_file url_of_pdf --structure_model_path TATR-v1.1-All/TATR-v1.1-All-msft.pth  --detection_model_path tatr-pubtables1m-v1.0/pubtables1m_detection_detr_r18.pth  -c
```

## Acknowledgements

This example is mostly adapted from [table-transformer](https://github.com/microsoft/table-transformer). We thank the related authors for their great work!
