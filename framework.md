# **Character Consistency Validator \- Architecture Framework**

## **1\. Project Goal**

A local web application to validate and curate datasets for LoRA training. It ensures facial identity and body consistency against a set of "Source of Truth" reference images.

## **2\. Technical Stack**

* **Architecture:** Local Web Server (Backend) \+ Browser Interface (Frontend).  
* **Frontend:** React (Vite) \+ Tailwind CSS \+ ShadcnUI.  
* **Backend:** Python (FastAPI) with venv for isolation.  
* **Communication:** HTTP/REST.  
  * *Development:* Vite Proxy forwards requests to FastAPI.  
  * *Production:* FastAPI serves the React static build.  
* **Database:** SQLite (local file, accessed via Python via SQLAlchemy).  
* **AI/ML:**  
  * *Face:* InsightFace.  
  * *Pose/Shape:* MMPose (RTMPose) \+ SMPL-X.  
  * *LLM:* Ollama (API compatible) \+ Google Gemini API.

## **3\. Core Constraints & Requirements**

* **Hardware Agnostic:** Must run on Apple Silicon (MPS) and Windows (CUDA).  
* **Memory Management:** **Strict Serialization.** The app must never load the Vision Models and the LLM into VRAM simultaneously. It must explicitly unload() one before load()ing the other.  
* **File Access:** Since this is a local web app, it operates on **Absolute File Paths**. The backend reads images directly from the disk. The Frontend displays them by requesting the backend to serve the file content or via a mounted static directory.  
* **Workflow:**  
  1. **Ingest:** User inputs Reference Image Paths \-\> Self-Validate \-\> Save Reference Profile.  
  2. **Process:** User inputs Dataset Folder Path \-\> Analyze (Vision) \-\> Store Metrics.  
  3. **Enrich:** Generate Captions (LLM) \-\> Store in DB.  
  4. **Curate:** Filter/Select based on scores.  
  5. **Export:** Backend copies files \+ writes .txt captions to a new Target Folder.

## **4\. Data Model (SQLite Schema Concept)**

* **Project:** id, name, lora\_preset\_type (SDXL, Flux, Face-Only).  
* **Character:** id, project\_id, name.  
* **ReferenceImage:** id, character\_id, path (absolute local path), view\_type, embedding\_blob, smpl\_params\_blob.  
* **DatasetImage:** id, character\_id, original\_path (absolute local path), status (pending, analyzed, rejected, approved).  
* **ImageMetrics:** image\_id, face\_similarity\_score, body\_consistency\_score, limb\_ratios\_json, shot\_type.  
* **Captions:** image\_id, model\_type (SDXL, Flux, Dense), text\_content.

## **5\. UI Layout Strategy**

* **Left Sidebar:** Project/Character navigation.  
* **Main Stage:**  
  * *Gallery View:* Grid of images with status indicators.  
  * *Detail View:* Split screen (Target vs Reference).  
* **Right Panel:**  
  * *Context:* Analysis results, sliders, and LLM Chat interface.