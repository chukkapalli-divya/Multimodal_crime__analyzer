# Multimodal Crime / Incident Report Analyzer

**Course:** AI for Engineers  
**Type:** Group Assignment (Group of 5)  
**Objective:** Build an AI-powered system that processes unstructured data from 5 different modalities and produces a unified structured incident report.

## Problem Statement

A city's emergency response department receives hundreds of incident reports daily from different sources — audio emergency calls, written police reports, CCTV footage, scene photographs, and social media posts. This project builds a prototype AI pipeline that automatically extracts structured information from each source and merges them into a single incident report for investigators.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Raw Unstructured Data                   │
│  Audio │ PDF Docs │ Images │ Video │ Text Posts          │
└───┬────────┬──────────┬────────┬────────┬───────────────┘
    │        │          │        │        │
    ▼        ▼          ▼        ▼        ▼
┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐
│ Whisper ││PyMuPDF ││ YOLOv8 ││ OpenCV ││ spaCy  │
│ spaCy  ││ spaCy  ││OpenCV  ││ YOLOv8 ││HugFace │
│HugFace ││        ││Tesser. ││PyTorch ││ NLTK   │
└───┬────┘└───┬────┘└───┬────┘└───┬────┘└───┬────┘
    │         │         │         │         │
    ▼         ▼         ▼         ▼         ▼
┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐
│Audio   ││Doc     ││Image   ││Video   ││Text    │
│CSV     ││CSV     ││CSV     ││CSV     ││CSV     │
│703 rows││10 rows ││5000row ││284 rows││115 rows│
└───┬────┘└───┬────┘└───┬────┘└───┬────┘└───┬────┘
    │         │         │         │         │
    └─────────┴─────┬───┴─────────┴─────────┘
                    ▼
        ┌───────────────────────┐
        │  Integration Module   │
        │  - Merge on INC_ID    │
        │  - Severity Scoring   │
        │  - Dashboard + Query  │
        └───────────┬───────────┘
                    ▼
        ┌───────────────────────┐
        │   Final Outputs       │
        │  - Unified CSV        │
        │  - Wide-Format CSV    │
        │  - Dashboard (PNG)    │
        └───────────────────────┘
```

## Repository Structure

```
multimodal-crime-analyzer/
├── README.md
├── requirements.txt
├── audio/
│   └── audio_analyst.ipynb          # Whisper transcription + NLP extraction
├── pdf/
│   └── document_analyst.ipynb       # PDF parsing + entity extraction
├── images/
│   └── image_analyst.ipynb          # YOLOv8 object detection + OCR
├── video/
│   └── video_analyst.ipynb          # Frame extraction + anomaly detection
├── text/
│   └── text_analyst.ipynb           # NER + sentiment + topic classification
├── integration/
│   └── integration.ipynb            # Merge all 5 outputs + dashboard
└── outputs/
    ├── audio_analyst_output.csv
    ├── document_analyst_output.csv
    ├── image_analyst_output.csv
    ├── video_analyst_output.csv
    ├── text_analyst_output.csv
    ├── final_merged_incidents.csv
    ├── final_unified_incidents.csv
    ├── final_integrated_wide.csv
    └── dashboard.png
```

## Individual Components

### 1. Audio Analyst
- **Input:** 911 emergency audio calls (WAV files)
- **Tools:** OpenAI Whisper, spaCy, HuggingFace Transformers
- **Process:** Speech-to-text transcription → keyword/entity extraction → sentiment & urgency analysis
- **Output:** `audio_analyst_output.csv` (703 calls) with columns: Call_ID, Transcript, Extracted_Event, Location, Sentiment, Urgency_Score
- **Dataset:** [911 Recordings — First 6 Seconds](https://www.kaggle.com/datasets/louisteitelbaum/911-recordings-first-6-seconds)

### 2. Document Analyst
- **Input:** Police department PDF reports (FOIA-released documents)
- **Tools:** PyMuPDF, pdfplumber, spaCy
- **Process:** PDF text extraction → NER for departments, dates, programs → structured output
- **Output:** `document_analyst_output.csv` (10 reports) with columns: Report_ID, Department, Doc_Type, Date, Program, Key_Detail
- **Dataset:** [Arkansas Police 1033 Training Proposals](https://www.muckrock.com/foi/arkansas-114/arkansas-police-departments-1033-training-plan-proposals-20493/#file-52365)

### 3. Image Analyst
- **Input:** Fire and smoke scene photographs
- **Tools:** YOLOv8 (Ultralytics), OpenCV, pytesseract
- **Process:** Object detection → scene classification → OCR for visible text
- **Output:** `image_analyst_output.csv` (5,000 images) with columns: Image_ID, Scene_Type, Objects_Detected, Bounding_Boxes, Confidence_Score
- **Dataset:** [Roboflow Fire Detection](https://universe.roboflow.com/search?q=fire)

### 4. Video Analyst
- **Input:** CCTV surveillance footage (CAVIAR dataset)
- **Tools:** OpenCV, YOLOv8, PyTorch
- **Process:** Frame extraction → motion/anomaly detection → event classification
- **Output:** `video_analyst_output.csv` (284 frames) with columns: Clip_ID, Timestamp, Frame_ID, Event_Detected, Persons_Count, Confidence_Score
- **Dataset:** [CAVIAR CCTV Dataset](https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/)

### 5. Text Analyst
- **Input:** Crime report text data (social media / news)
- **Tools:** spaCy, HuggingFace Transformers, NLTK
- **Process:** Text preprocessing → NER → sentiment analysis → topic classification
- **Output:** `text_analyst_output.csv` (115 reports) with columns: Text_ID, Crime_Type, Location_Entity, Sentiment, Topic, Severity_Label
- **Dataset:** [CrimeReport — Kaggle](https://www.kaggle.com/datasets/cameliasiadat/crimereport)

## Integration

The integration notebook (`integration/integration.ipynb`) performs:

1. **Load** all 5 analyst CSVs
2. **Standardize** column names across modalities
3. **Assign** common Incident_IDs and merge using `pandas`
4. **Handle** missing values with `fillna`
5. **Classify** severity (Low / Medium / High) based on event type, confidence, and sentiment
6. **Visualize** with a 4-panel dashboard (source distribution, severity breakdown, top events, confidence scores)
7. **Query interface** to filter by severity, modality, event type, or confidence threshold

### Final Integrated Output Structure

| Incident_ID | Audio_Event | PDF_Doc_Type | Image_Objects | Video_Event | Text_Crime_Type | Severity |
|-------------|-------------|--------------|---------------|-------------|-----------------|----------|
| INC_001     | Assault     | 1033 Training Proposal | fire, smoke | Person walking | Robbery | High |

## How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/multimodal-crime-analyzer.git
   cd multimodal-crime-analyzer
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run individual analyst notebooks** (recommended on Google Colab or Kaggle for GPU access):
   - Each notebook in `audio/`, `pdf/`, `images/`, `video/`, `text/` can be run independently
   - Output CSVs are saved to `outputs/`

4. **Run the integration notebook:**
   - Upload all 5 output CSVs to the same environment
   - Run `integration/integration.ipynb` to generate the merged dataset and dashboard

## Technologies Used

| Category | Tools |
|----------|-------|
| Speech-to-Text | OpenAI Whisper |
| NLP / NER | spaCy, NLTK, HuggingFace Transformers |
| Object Detection | YOLOv8 (Ultralytics) |
| Computer Vision | OpenCV, pytesseract (OCR) |
| PDF Processing | PyMuPDF, pdfplumber |
| Data Processing | pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Deep Learning | PyTorch, TorchVision |
