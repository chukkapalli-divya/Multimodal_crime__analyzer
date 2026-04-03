<div align="center">

# 🔍 Multimodal Crime / Incident Report Analyzer

### AI-Powered Emergency Incident Analysis System

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)](https://ultralytics.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![Whisper](https://img.shields.io/badge/OpenAI-Whisper-412991?style=for-the-badge&logo=openai&logoColor=white)](https://github.com/openai/whisper)

| Course | Type |
|--------|------|
| **AI for Engineers** | **Group of 5 Students** |

[Background](#1--background-story) •
[Pipeline](#2--expected-ai-pipeline) •
[Roles](#3--individual-student-roles--tasks) •
[Integration](#4--final-integration-task) •
[Deliverables](#5--deliverables) •
[Setup](#-installation--setup) •
[Collaboration](#-team-collaboration)

---

</div>

## 1. 📖 Background Story

A city's emergency response department is facing a major challenge. Every day, hundreds of incidents such as road accidents, thefts, fires, and public disturbances are reported from different sources. These reports come in many formats:

| Source | Format | Example |
|--------|--------|---------|
| 🎙️ Emergency Calls | Audio (WAV) | 911 caller reporting a fire |
| 📄 Police Reports | PDF Documents | Official incident filing |
| 📸 Scene Photos | Images (JPG/PNG) | CCTV snapshot of fire/smoke |
| 🎥 Surveillance | Video (MPG/MP4) | CCTV footage of altercation |
| 📝 Witness Posts | Text (CSV/JSON) | Social media crime report |

Currently, human analysts must manually review all this information to understand what actually happened during an incident. This process is **slow, error-prone, and makes it difficult to respond quickly to emergencies**.

> **Our Solution:** An AI-powered Multimodal Incident Analyzer that automatically collects information from different types of unstructured data and converts it into a structured incident report that investigators can easily analyze.

---

## 2. 🔄 Expected AI Pipeline

All five components connect into one unified pipeline:

```
╔══════════════════════════════════════════════════════════════════════════════╗
║  STAGE 1: UNSTRUCTURED DATA INGESTION                                      ║
║  Audio files, PDFs, images, video clips, and text posts are collected      ║
╚══════════════════════════════╤═══════════════════════════════════════════════╝
                               ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  STAGE 2: AI PROCESSING PER MODALITY                                       ║
║                                                                            ║
║  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    ║
║  │ 🎙️ Audio │  │ 📄 PDF   │  │ 📸 Image │  │ 🎥 Video │  │ 📝 Text  │    ║
║  │          │  │          │  │          │  │          │  │          │    ║
║  │ Whisper  │  │ PyMuPDF  │  │ YOLOv8   │  │ OpenCV   │  │ spaCy    │    ║
║  │ spaCy    │  │ spaCy    │  │ OpenCV   │  │ YOLOv8   │  │ HugFace  │    ║
║  │ HugFace  │  │ pdfplumb │  │ Tesseract│  │ PyTorch  │  │ NLTK     │    ║
║  └─────┬────┘  └─────┬────┘  └─────┬────┘  └─────┬────┘  └─────┬────┘    ║
║        │             │             │             │             │          ║
╚════════╪═════════════╪═════════════╪═════════════╪═════════════╪══════════╝
         ▼             ▼             ▼             ▼             ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  STAGE 3: INFORMATION EXTRACTION                                           ║
║  Key fields extracted: event, location, time, entities, sentiment          ║
║                                                                            ║
║  audio.csv    doc.csv     image.csv    video.csv    text.csv              ║
║  (703 rows)   (10 rows)   (5000 rows)  (284 rows)   (115 rows)           ║
╚══════════════════════════╤═══════════════════════════════════════════════════╝
                           ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  STAGE 4: STRUCTURED DATASET GENERATION                                    ║
║  All extracted outputs merged into unified CSV using pandas merge/join     ║
║  Missing values handled with fillna("N/A")                                ║
║  Severity classification: Low / Medium / High                             ║
╚══════════════════════════╤═══════════════════════════════════════════════════╝
                           ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  STAGE 5: DASHBOARD / QUERY SYSTEM                                         ║
║  4-panel matplotlib dashboard + Python query interface                     ║
║  Filter by: severity, modality, event type, confidence score              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### Final Integrated Output Structure

| Incident_ID | Source | Event | Location | Time | Severity |
|-------------|--------|-------|----------|------|----------|
| INC_001 | Audio + PDF | Road Accident | Main St. | 14:32 | High |

---

## 3. 🧑‍💻 Individual Student Roles & Tasks

Each student owns one data modality end-to-end — from ingestion to structured output. All five outputs are merged in the final integration task.

---

### Keerthana Gummuluri Venkata — 🎙️ Audio Analyst

**Data Type:** Emergency audio calls / witness voice statements

| Item | Details |
|------|---------|
| **Dataset** | [911 Recordings — First 6 Seconds](https://www.kaggle.com/datasets/louisteitelbaum/911-recordings-first-6-seconds) |
| **Records** | 703 audio calls |
| **Tools** | `openai-whisper` · `spaCy` · `transformers (HuggingFace)` |
| **Notebook** | `audio/audio_analyst.ipynb` |

**Tasks:**
- Convert audio files to text using Whisper speech-to-text model
- Extract keywords: incident type, location mentions, names, urgency phrases
- Perform sentiment/urgency analysis on transcribed text (calm vs distressed)
- Output structured CSV

**Output Schema:**

| Column | Example | Description |
|--------|---------|-------------|
| `Call_ID` | C001 | Unique identifier |
| `Transcript` | *There is a fire, people are trapped...* | Whisper transcription |
| `Extracted_Event` | *Building fire / trapped persons* | Keyword-classified incident |
| `Location` | *Downtown Ave* | spaCy NER extracted location |
| `Sentiment` | *Distressed* | Calm / Concerned / Distressed |
| `Urgency_Score` | *0.91* | 0.0–1.0 weighted score |

---

### Varalaxmi Jangili — 📄 Document Analyst

**Data Type:** Police reports / official incident documents (PDF)

| Item | Details |
|------|---------|
| **Dataset** | [Arkansas Police 1033 Training Proposals](https://www.muckrock.com/foi/arkansas-114/arkansas-police-departments-1033-training-plan-proposals-20493/#file-52365) |
| **Records** | 10 PDF reports |
| **Tools** | `PyMuPDF (fitz)` · `pdfplumber` · `pytesseract` · `spaCy` |
| **Notebook** | `pdf/document_analyst.ipynb` |

**Tasks:**
- Extract raw text from PDFs using PDF parsing library
- Identify and extract: incident type, date, location, officer name, suspect description, outcome
- Handle both text-based and scanned PDFs (OCR for scanned)
- Output structured CSV

**Output Schema:**

| Column | Example | Description |
|--------|---------|-------------|
| `Report_ID` | RPT_001 | Unique identifier |
| `Department` | *Arkansas PD* | Police department |
| `Doc_Type` | *1033 Training Proposal* | Document classification |
| `Date` | *2015-04-10* | Extracted date |
| `Program` | *Law Enforcement Support* | Program name |
| `Key_Detail` | *Equipment request: tactical gear listed* | Key information summary |

---

### Abhimanyu Raj put — 📸 Image Analyst

**Data Type:** Crime scene / accident scene photographs

| Item | Details |
|------|---------|
| **Dataset** | [Roboflow Fire Detection](https://universe.roboflow.com/search?q=fire) |
| **Records** | 5,000 images |
| **Tools** | `YOLOv8 (ultralytics)` · `OpenCV` · `pytesseract` · `torchvision` |
| **Notebook** | `images/image_analyst.ipynb` |

**Tasks:**
- Run object detection to identify: vehicles, fire, people, weapons, damage
- Classify scene type: accident, fire, theft, public disturbance
- Extract visible text using OCR (license plates, street signs)
- Output structured CSV

**Output Schema:**

| Column | Example | Description |
|--------|---------|-------------|
| `Image_ID` | IMG_034 | Unique identifier |
| `Scene_Type` | *Fire Scene* | Scene classification |
| `Objects_Detected` | *fire, smoke* | Detected objects |
| `Bounding_Boxes` | *2 fire regions, 1 smoke plume* | Detection regions |
| `Text_Extracted` | — | OCR-extracted text |
| `Confidence_Score` | *0.94* | Detection confidence |

---

### Divya Chukkapalli — 🎥 Video Analyst

**Data Type:** CCTV / surveillance footage

| Item | Details |
|------|---------|
| **Dataset** | [CAVIAR CCTV Dataset](https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/) |
| **Records** | 284 frames from 5 video clips |
| **Tools** | `OpenCV` · `YOLOv8` · `PyTorch` · `imageio / moviepy` |
| **Notebook** | `video/video_analyst.ipynb` |

**Tasks:**
- Extract frames from video clips at regular intervals
- Apply motion detection or anomaly detection to identify events
- Detect and classify objects/activities: running, fighting, vehicle movement, fire
- Output structured CSV

**Output Schema:**

| Column | Example | Description |
|--------|---------|-------------|
| `Clip_ID` | CAVIAR_03 | Source video |
| `Timestamp` | *00:00:12* | Frame timestamp |
| `Frame_ID` | FRM_036 | Frame identifier |
| `Event_Detected` | *Person collapsing* | Classified event |
| `Persons_Count` | *1 person* | Detected persons |
| `Confidence_Score` | *0.88* | Detection confidence |

---

### Ruchith Reddy Parnem — 📝 Text Analyst

**Data Type:** Social media posts / news articles

| Item | Details |
|------|---------|
| **Dataset** | [CrimeReport — Kaggle](https://www.kaggle.com/datasets/cameliasiadat/crimereport) |
| **Records** | 115 crime reports |
| **Tools** | `spaCy` · `transformers (HuggingFace)` · `NLTK` · `pandas` |
| **Notebook** | `text/text_analyst.ipynb` |

**Tasks:**
- Clean and preprocess raw text: remove noise, normalize, tokenize
- Run NER to extract: people, locations, organizations, dates
- Perform sentiment analysis and topic classification
- Output structured CSV

**Output Schema:**

| Column | Example | Description |
|--------|---------|-------------|
| `Text_ID` | TXT_112 | Unique identifier |
| `Crime_Type` | *Robbery* | Crime classification |
| `Location_Entity` | *Oak Street, Chicago* | Extracted location |
| `Sentiment` | *Negative* | Sentiment label |
| `Topic` | *Theft / Robbery* | Topic category |
| `Severity_Label` | *High* | Severity classification |

---

## 4. 🔗 Final Integration Task

After each student completes their individual component, the full team combines all five structured outputs into a single unified incident dataset.

### Integration Steps

| Step | Action | Implementation |
|------|--------|----------------|
| **Step 1** | Define common Incident_ID | Sequential `INC_001`, `INC_002`... across all 5 CSVs |
| **Step 2** | Merge all 5 DataFrames | `pandas.merge()` / `join()` on Incident_ID |
| **Step 3** | Handle missing values | `fillna("N/A")` where a modality has no data |
| **Step 4** | Severity classification | Low / Medium / High based on combined signals |
| **Step 5** | Dashboard / Query system | 4-panel matplotlib charts + Python filter interface |

### Final Integrated Dataset Structure

| Incident_ID | Audio_Event | PDF_Doc_Type | Image_Objects | Video_Event | Text_Crime_Type | Severity |
|-------------|-------------|--------------|---------------|-------------|-----------------|----------|
| INC_001 | *Building fire / trapped* | *1033 Training Proposal* | *fire, smoke (0.94)* | *Person collapsing* | *Robbery / Theft* | **High** |

### Severity Classification Rules

| Severity | Trigger Keywords | Description |
|----------|-----------------|-------------|
| 🔴 **High** | fire, shooting, assault, trapped, collapse, explosion, weapon, stabbing | Immediate response required |
| 🟡 **Medium** | accident, theft, robbery, burglary, collision, injured, fight | Significant incident |
| 🟢 **Low** | suspicious, disturbance, noise, walking, normal, minor, vandalism | Routine / minor event |

### Dashboard Visualizations

The integration notebook generates a 4-panel dashboard:

| Panel | Chart Type | Shows |
|-------|-----------|-------|
| **Top-Left** | Bar chart | Incidents by data source (Audio/PDF/Image/Video/Text) |
| **Top-Right** | Pie chart | Severity distribution (High/Medium/Low percentages) |
| **Bottom-Left** | Horizontal bar | Top 8 event types across all modalities |
| **Bottom-Right** | Histogram | Confidence score distribution per modality |

## 📁 Repository Structure

```
Multimodal-Incident-Analyzer/
│
├── 📄 README.md                          # Project documentation
├── 📄 requirements.txt                   # Python dependencies
│
├── 🎙️ audio/
│   └── audio_analyst.ipynb               # Whisper + spaCy + HuggingFace
│
├── 📄 pdf/
│   └── document_analyst.ipynb            # PyMuPDF + pdfplumber + spaCy
│
├── 📸 images/
│   └── image_analyst.ipynb               # YOLOv8 + OpenCV + pytesseract
│
├── 🎥 video/
│   └── video_analyst.ipynb               # OpenCV + YOLOv8 + PyTorch
│
├── 📝 text/
│   └── text_analyst.ipynb                # spaCy + HuggingFace + NLTK
│
├── 🔗 integration/
│   └── integration.ipynb                 # Merge + Severity + Dashboard
│
└── 📊 outputs/
    ├── audio_analyst_output.csv          # 703 transcribed 911 calls
    ├── document_analyst_output.csv       # 10 parsed police PDF reports
    ├── image_analyst_output.csv          # 5,000 fire/smoke detected scenes
    ├── video_analyst_output.csv          # 284 CCTV analyzed frames
    ├── text_analyst_output.csv           # 115 classified crime reports
    ├── final_merged_incidents.csv        # Cross-modal merged dataset
    ├── final_unified_incidents.csv       # Long-format unified dataset
    ├── final_integrated_wide.csv         # Wide-format (assignment spec)
    └── dashboard.png                     # 4-panel visualization
```

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.10+
- pip
- Git
- Google Colab or Kaggle account (recommended for GPU access)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/chukkapalli-divya/Multimodal_crime__analyzer.git
cd Multimodal-Incident-Analyzer

# Install all dependencies
pip install -r requirements.txt

# Download spaCy English model
python -m spacy download en_core_web_sm
```

### Running the Notebooks

| Notebook | Platform | GPU | Run Command |
|----------|----------|-----|-------------|
| `audio/audio_analyst.ipynb` | Kaggle | ✅ Required | Upload to Kaggle → Enable GPU → Run All |
| `pdf/document_analyst.ipynb` | Colab | ❌ Not needed | Upload to Colab → Run All |
| `images/image_analyst.ipynb` | Colab | ✅ Required | Upload to Colab → Enable T4 GPU → Run All |
| `video/video_analyst.ipynb` | Colab | ✅ Required | Upload to Colab → Enable T4 GPU → Run All |
| `text/text_analyst.ipynb` | Colab | ❌ Not needed | Upload to Colab → Run All |
| `integration/integration.ipynb` | Colab | ❌ Not needed | Upload 5 CSVs + notebook → Run All |

### Query the Final Dataset

```python
# All HIGH severity incidents
query_incidents(df_unified, severity="High")

# All fire-related incidents
query_incidents(df_unified, event_type="fire")

# Audio modality only
query_incidents(df_unified, modality="Audio")

# High-confidence detections (>= 0.9)
query_incidents(df_unified, min_confidence=0.9)
```

---

## 🛠️ Technologies Used

| Category | Tools and Libraries |
|----------|-------------------|
| **Speech-to-Text** | OpenAI Whisper |
| **NLP & NER** | spaCy (`en_core_web_sm`), NLTK |
| **Sentiment Analysis** | HuggingFace Transformers (DistilBERT SST-2) |
| **Object Detection** | YOLOv8 (Ultralytics) |
| **Computer Vision** | OpenCV |
| **OCR** | pytesseract |
| **PDF Processing** | PyMuPDF (fitz), pdfplumber |
| **Deep Learning** | PyTorch, TorchVision, TorchAudio |
| **Data Processing** | pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Video Processing** | OpenCV, imageio, moviepy |

---

## 📚 Datasets

| Modality | Dataset | Source | Access | Size |
|----------|---------|--------|--------|------|
| 🎙️ Audio | 911 Recordings — First 6 Seconds | [Kaggle](https://www.kaggle.com/datasets/louisteitelbaum/911-recordings-first-6-seconds) | Sign in → Download | 703 WAV files |
| 📄 Document | Arkansas Police 1033 Training Proposals | [MuckRock](https://www.muckrock.com/foi/arkansas-114/arkansas-police-departments-1033-training-plan-proposals-20493/#file-52365) | Direct download, no account | 1 PDF (10 reports) |
| 📸 Image | Roboflow Fire Detection | [Roboflow](https://universe.roboflow.com/search?q=fire) | Free Roboflow account | 5,000+ images |
| 🎥 Video | CAVIAR CCTV Dataset | [Edinburgh](https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/) | Direct download, no account | 5 MPG clips |
| 📝 Text | CrimeReport | [Kaggle](https://www.kaggle.com/datasets/cameliasiadat/crimereport) | Sign in → Download | 115 text records |

---

## 🤝 Team Collaboration

### Collaboration Workflow

```
        ┌──────────────────────────────────────────────────┐
        │              TEAM EFFORTS                  │
        │  Creates repo → Folder structure → Integration    │
        └─────────────────────┬────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │          │          │          │           │
   ┌────┴───┐ ┌───┴────┐ ┌───┴────┐ ┌───┴────┐ ┌───┴────┐
   │Keerthana Gummuluri Venkata│ │Varalaxmi Jangili│ │Abhimanyu Raj put│ │Divya Chukkapalli│ │Ruchith Reddy Parnem│
   │ Audio  │ │  PDF   │ │ Image  │ │ Video  │ │  Text  │
   └────┬───┘ └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘
        │         │          │          │           │
        │    Each member develops independently     │
        │    in their own folder                    │
        │         │          │          │           │
        └─────────┴─────┬────┴──────────┴───────────┘
                        ▼
              ┌──────────────────┐
              │  INTEGRATION     │
              │  (Team Effort)   │
              │  Merge 5 CSVs → │
              │  Dashboard →     │
              │  Final Output    │
              └──────────────────┘
```


## 👥 Team

| Role | Member | GitHub | Contribution |
|------|--------|--------|-------------|
| 🎙️ Audio Analyst | Keerthana Gummuluri Venkata | [@keerthana200401](https://github.com/keerthana200401]) | Whisper transcription + spaCy NER + HuggingFace sentiment |
| 📄 Document Analyst | Varalaxmi Jangili | [@varalaxmijangili12](https://github.com/varalaxmijangili12) | PDF parsing + entity extraction + OCR |
| 📸 Image Analyst | Abhimanyu Raj put| [@Abhimanyu1801](https://github.com/Abhimanyu1801) | YOLOv8 object detection + scene classification + OCR |
| 🎥 Video Analyst | Divya Chukkapalli | [@chukkapalli-divya](https://github.com/chukkapalli-divya) | Frame extraction + motion detection + anomaly classification |
| 📝 Text Analyst | Ruchith Reddy Parnem | [@Ruchith1508](https://github.com/Ruchith1508) | NER + sentiment analysis + topic classification |
| 🔗 Integration | All Members | — | Merge + severity scoring + dashboard + query interface |

---
