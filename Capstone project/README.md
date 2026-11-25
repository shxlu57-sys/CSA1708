# AI-Powered Medical Report Analysis and Patient Health Insight System
CSA1708 - Artificial Intelligence | SIMATS Engineering

Upload a hospital report (PDF / photo) and get simple, large-font, colour-coded
explanations in English, Telugu, Hindi or Tamil.

## Setup (VS Code)
1. Install Python 3.9+ and Tesseract OCR
   (Windows: https://github.com/UB-Mannheim/tesseract/wiki - default path is used automatically).
2. Open this folder in VS Code, then in the terminal:
       pip install -r requirements.txt
3. Run:
       python main.py

## Folder
- main.py                      - complete application (OCR, analysis, ML, GUI)
- requirements.txt             - Python libraries
- sample_reports/sample_report.txt - sample text to test (or click "Try Sample Report")

## Modules (as in the presentation)
1. Upload & Data Extraction - PyPDF2 / PyMuPDF, OpenCV, Tesseract, Pandas
2. AI-Based Analysis        - reference ranges + Scikit-learn risk model (demo, synthetic data)
3. Health Insight & Report  - Tkinter GUI, 4 languages, adjustable font size, save as TXT/CSV

Note: AI helper only, not a medical diagnosis. Always consult a doctor.
