# VietTable-Transformer
A pipe line for recognize and OCR VietNam scan table 

This project provides a pipeline to:
- Convert **PDF files into images**.
- Detect **tables in documents** using YOLO.
- Recognize **text inside tables** with VietOCR.
- Export results into a **CSV file**.


## Installation

### Clone repository & create environment
```bash
git clone https://github.com/Azinale/VietTable-Transformer.git
cd ./VietTB_transformer
pip install -r requirements.txt

ocr_project/
│── main.py                # Main pipeline entry point
│── requirements.txt
│── README.md
│
├── configs/
│   └── settings.py        # Global configuration
│
├── models/
│   ├── yolo_wrapper.py    # Load & detect with YOLO
│   ├── ocr_easyocr.py     # OCR using EasyOCR
│
├── processing/
│   ├── pdf_utils.py       # PDF → images
│   ├── image_utils.py     # Image processing utilities
│   └── table_extractor.py # Combine YOLO + OCR
│
├── utils/
│   ├── timer.py           # Timing utility
│   └── file_io.py         # File dialog & CSV saving
│
└── outputs/               # OCR results
```
### How to run?
```bash
python main.py
```
