# ANLP-End-to-End-RAG-System
Github repository for Assignment 2 (11711 - Advanced Natural Language Processing (Fall 2024))

## Team Members
- Akshita Gupta (akshita3@andrew.cmu.edu)
- Krishnaprasad Vijayshankar (kvijaysh@andrew.cmu.edu)
- Mahita Kandala (mkandala@andrew.cmu.edu)

## Github Structure
```
ANLP-End-to-End-RAG-System/
├── data/
│   ├── test/
│   │   ├── questions.txt
│   │   └── reference_answers.txt
│   └── train/
│       ├── questions.txt
│       └── reference_answers.txt
├── models/
├── notebooks/
│   ├── annotate/
│   ├── data_scraping/
│   └── models/
├── .gitignore
├── README.md
├── requirements.txt
├── rag.py
└── text_files/
```

## Setup
```bash
pip install -r requirements.txt
```

Ollama server should be running in the background.(https://ollama.ai/download)

Ollama setup instructions:
```bash
ollama pull llama3.2
ollama serve
```

## How to run the code

The code is designed to run for both single question and entire question set. 
It allows for zero-shot and few-shot prompting.

To run for a single question, run the following command:

 - Zero-shot:
    ```bash
    python rag.py --data_dir <data_dir> --query <query> --zero_shot
    ```

 - Few-shot:
    ```bash
    python rag.py --data_dir <data_dir> --query <query>
    ```

To run for the entire question set, run the following command:

 - Zero-shot:
    ```bash
    python rag.py --data_dir <data_dir> --input <input> --output <output> --zero_shot
    ```

 - Few-shot:
    ```bash
    python rag.py --data_dir <data_dir> --input <input> --output <output>
    ```

Note: The --force_vs flag can be used to force the creation of a new vector store instead of using cached version
