# Manufacturing Insight Copilot (Python)

A multi-agent analytics system built with **Streamlit** and **Google GenAI SDK**.

## Features
- **DataAgent**: Performs heavy statistical profiling (Code Execution Tool) locally.
- **InsightAgent**: Interprets trends and statistics using Gemini 2.0 Flash.
- **KnowledgeAgent**: Retrieves relevant SOPs (Simulated RAG).
- **ReportAgent**: Generates Technical and Executive markdown reports.

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **API Key**
   - You need a Google Cloud API Key with access to Gemini.
   - You can enter it in the UI or create a `.env` file:
     ```
     GOOGLE_API_KEY=your_key_here
     ```

3. **Run the App**
   ```bash
   streamlit run app.py
   ```

## Architecture
- `app.py`: Main UI and Orchestrator.
- `tools/data_analysis.py`: Local Pandas-based processing (No raw data sent to LLM).
- `agents/`: Contains logic for specific agent roles.
- `core/`: Data models and config.
