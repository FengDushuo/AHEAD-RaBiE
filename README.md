# AHEAD-RaBiE  
**Artificial High-performance Enzyme Acceleration Design**

AHEAD-RaBiE is a research-oriented software platform designed to support enzyme engineering and catalytic system studies using artificial intelligence techniques.  
The platform integrates literature knowledge management, large language model (LLM) agents, machine learning (ML) prediction models, and density functional theory (DFT) result management into a unified computational framework.

This repository provides the **core source code and deployment logic** of the platform, with a strong emphasis on **environment configuration, executability, and reproducibility**.

---

## 1. Software Scope

This software is intended as a **research assistance tool**, suitable for:

- Enzyme engineering and nanozyme research  
- Computational catalysis and materials informatics  
- AI-assisted literature analysis and knowledge discovery  
- Machine learning–based approximation of DFT properties  

The software does **not replace experimental validation**. All outputs should be interpreted as computational references.

---

## 2. System Architecture

AHEAD-RaBiE adopts a **backend–frontend separation architecture**:

- **Backend (Python-based)**  
  Handles data processing, ML inference, LLM interaction, and API services.

- **Frontend (Web-based)**  
  Provides user interaction and visualization via a browser.

- **Database (MySQL)**  
  Stores user authentication information and structured metadata.

---

## 3. Hardware Requirements

### Minimum
| Component | Requirement |
|--------|-------------|
| CPU | ≥ 2 cores |
| Memory | ≥ 2 GB |
| Disk | ≥ 50 GB (excluding datasets) |

### Recommended
| Component | Recommendation |
|--------|----------------|
| CPU | ≥ 8 cores |
| Memory | ≥ 16 GB |
| Disk | ≥ 100 GB |
| GPU | Optional (recommended for ML tasks) |

---

## 4. Operating System Support

- **Linux (recommended)**  
  - Ubuntu 20.04+  
  - CentOS 7 / Rocky Linux 8  

- **Windows**  
  - Windows 8 or later (development/testing only)

Linux is strongly recommended for production deployment.

---

## 5. Software Dependencies

### Core Environment
| Component | Version |
|--------|--------|
| Python | ≥ 3.8 |
| MySQL | 8.0.27 |
| Browser | Chrome / Microsoft Edge |

### Create Python Environment (Recommended)
A dedicated Python environment is strongly recommended.

Using Conda:
```bash
conda create -n ahead python=3.8
conda activate ahead
```

### Python Libraries (Typical)
Install required Python packages: pip install -r requirements.txt
```text
datasets==3.6.0
flashgeotext==0.5.5
gensim==4.3.0
gensim==3.8.3
joblib==1.3.2
joblib==1.2.0
joblib==1.0.1
joblib==1.4.2
keybert==0.8.5
matplotlib==3.5.3
matplotlib==3.5.1
matplotlib==3.0.3
matplotlib==3.3.4
matplotlib==3.7.5
multi_rake==0.0.2
networkx==2.3
networkx==3.4.2
networkx==3.0
networkx==3.3
networkx==2.4
networkx==2.5
nltk==3.6.1
nltk==3.9.1
numpy==1.21.6
numpy==1.23.2
numpy==1.22.2
numpy==1.18.5
numpy==1.19.5
numpy==1.15.1
numpy==1.20.1
numpy==1.21.5
numpy==1.24.4
pandas==1.2.1
pandas==1.5.2
pandas==1.2.4
pandas==1.3.5
pandas==2.0.3
Pillow==9.1.0
Pillow==9.2.0
Pillow==9.0.1
Pillow==8.4.0
Pillow==9.3.0
Pillow==8.2.0
Pillow==11.3.0
pmdarima==2.0.4
pyecharts==1.9.1
pyecharts==2.0.6
PyMySQL==1.0.2
rake_nltk==1.0.6
scikit_learn==1.0.2
scikit_learn==1.2.0
scikit_learn==1.1.2
scikit_learn==0.24.1
scikit_learn==1.3.2
scipy==1.7.3
scipy==1.9.1
scipy==1.6.2
seaborn==0.11.1
seaborn==0.10.1
spacy==3.7.5
statsmodels==0.13.2
statsmodels==0.12.2
statsmodels==0.14.1
torch==1.13.1
torch==2.6.0
torch==2.3.1
torch==2.4.1
tornado==6.2
tornado==6.1
tqdm==4.67.0
tqdm==4.64.1
tqdm==4.66.5
tqdm==4.67.1
tqdm==4.66.2
tqdm==4.59.0
transformers==4.49.0
transformers==4.45.2
wordcloud==1.9.4
XlsxWriter==3.2.0
XlsxWriter==1.3.8
```

## 6. Database Configuration (MySQL)

### 6.1 Install MySQL

Install and start **MySQL 8.0.27** on your system.  
Ensure the MySQL service is running before starting the backend.

### 6.2 Create the Required Database

Create the database used by the backend service:

```sql
CREATE DATABASE login_db DEFAULT CHARSET utf8mb4;
```
### 6.3 Configure Database Connection
Configure database connection parameters in the backend configuration file:
```sql
mysqluser = "root"
mysqlpassword = "your_password"
hostsrc = "localhost"
port = 3306
```

## 7. Installation
Clone the repository and enter the project directory:
```text
git clone https://github.com/FengDushuo/AHEAD-RaBiE.git
cd AHEAD-RaBiE
```
Before proceeding, ensure that:
-The Python environment is activated
-All dependencies are installed
-The MySQL service is running

The following manifest will download three zip files to static/data/, unzip them, and automatically restore them to:
```text
static/data/db/
static/data/history/
static/data/nanoenzyme-medline-102000-20250706_1751862932/
static/data/checkpoint-10000-merged/
```
manifest.json:
```text
{
    "dataset_name": "AHEAD-RaBiE static/data bundle",
    "dataset_version": "2025-12-28",
    "zenodo_record_id": "18076202",
    "base_dir_static": "static/data",
    "base_dir_upload": "upload",
    "files": [
      {
        "target": "static",
        "path": "db.zip",
        "url": "https://zenodo.org/records/18076202/files/db.zip?download=1",
        "sha256": "a80f9f73f6067ecaf17e96f64e8b43b5ad6e25ab2fdae6c86b4121f34a01619c",
        "unpack": "zip"
      },
      {
        "target": "static",
        "path": "history.zip",
        "url": "https://zenodo.org/records/18076202/files/history.zip?download=1",
        "sha256": "afae34716db77af3862b2a7b8c40564b55204303c2fb28811a569e8a88c3b586",
        "unpack": "zip"
      },
      {
        "target": "static",
        "path": "nanoenzyme-medline-102000-20250706_1751862932.zip",
        "url": "https://zenodo.org/records/18076202/files/nanoenzyme-medline-102000-20250706_1751862932.zip?download=1",
        "sha256": "144047e585d61725397f81f6f47fd5220fd25ea2bfcc3d77813b6350ede82457",
        "unpack": "zip"
      }，
      {
        "target": "static",
        "path": "checkpoint-10000-merged.zip",
        "url": "https://zenodo.org/records/18076202/files/checkpoint-10000-merged.zip?download=1",
        "sha256": "0e63e901908a8659371417d705a80c206d455de50711b951d9e8987f88fc5125",
        "unpack": "zip"
      }
    ]
  }
```
Download database from zenodo:
```bash
python static/scripts/download_data.py --manifest static/data/manifest.json
```
Download into static/data/ and decompress.

## 8. Running the System
### 8.1 Local or Server Execution
Run the LLM：
```bash
bash static/data/llm-app-Llama-31-8B-finetune.sh
```
Run the backend service from the project root directory:
```bash
python server.py --port 8000
```

If the service starts successfully, it will listen on the specified port.
Access the system via:
-Local access:
http://127.0.0.1:8000
-Server access:
http://<server-ip>:8000

### 8.2 Changing the Service Port
```bash
python server.py --port 8001
```
This is useful when multiple instances are running on the same machine.
