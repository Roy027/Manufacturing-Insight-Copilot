import { GoogleGenAI, Type } from "@google/genai";
import { DataSummary, AnalysisReport } from "../types";
import { KNOWLEDGE_BASE_SOPS } from "../constants";

const getAiClient = () => {
  const apiKey = process.env.API_KEY;
  if (!apiKey) {
    throw new Error("API Key not found in process.env");
  }
  return new GoogleGenAI({ apiKey });
};

// 1. DataAgent: Replaced by the Code Execution Tool (dataProcessing.ts), 
// but we perform a "Task Handover" here if we wanted to summarize the summary.
// For the spec, InsightAgent consumes the profile directly.

// 2. InsightAgent
export const generateInsights = async (summary: DataSummary): Promise<string> => {
  const ai = getAiClient();
  
  // --- SECURITY & TOKEN BUDGET CHECK ---
  // 1. EXCLUDE RAW DATA: Explicitly strip 'raw_data_subset' so strictly NO raw rows are sent.
  // 2. PRUNE COLUMNS: Limit profiles to top 15 columns to prevent huge token usage on wide datasets.
  const MAX_CONTEXT_COLS = 15;
  
  const numericKeys = Object.keys(summary.numeric_profile).slice(0, MAX_CONTEXT_COLS);
  const prunedNumericProfile = Object.fromEntries(
    numericKeys.map(k => [k, summary.numeric_profile[k]])
  );

  const catKeys = Object.keys(summary.categorical_profile).slice(0, MAX_CONTEXT_COLS);
  const prunedCategoricalProfile = Object.fromEntries(
    catKeys.map(k => [k, summary.categorical_profile[k]])
  );

  const context = {
    task: "generate_technical_insights",
    data_profile: {
      n_rows: summary.n_rows,
      n_cols: summary.n_cols,
      // Only send the pruned statistical profiles
      numeric_profile: prunedNumericProfile,
      categorical_profile: prunedCategoricalProfile,
      top_correlations: summary.top_correlations, // Already capped at 20 in dataProcessing
      time_profiles: summary.time_profiles,
      anomalies: summary.anomalies,
      // Only send the small 5-row sample, NEVER the full dataset
      sample: summary.sample_rows
    }
  };

  const prompt = `
    You are the InsightAgent, an expert in manufacturing data analytics.
    
    You are provided with a 'Code Execution Tool' output containing a statistical profile of the dataset.
    DO NOT request raw data. Use the provided statistics, trends, and correlations.

    Input Data Profile:
    ${JSON.stringify(context, null, 2)}
    
    Task:
    1. Analyze trends in the 'time_profiles'. Are there degradations?
    2. Interpret 'top_correlations'. Do they indicate physical relationships (e.g. Temp vs Pressure)?
    3. Evaluate 'anomalies'. Is the dataset stable or noisy?
    
    Output:
    Provide a list of key technical findings, hypotheses, and potential root causes.
    Focus on deviations from normality.
  `;

  const response = await ai.models.generateContent({
    model: 'gemini-2.5-flash',
    contents: prompt,
  });

  return response.text || "No insights generated.";
};

// 3. KnowledgeAgent
export const retrieveKnowledge = async (currentInsights: string): Promise<string> => {
  const ai = getAiClient();
  
  const prompt = `
    You are the KnowledgeAgent. You have access to the company's SOPs and Historical Cases (Long-term Memory).
    
    Your Task:
    Read the 'Current Insights' and find relevant documents in the 'Knowledge Base'.
    Map specific observed problems to SOPs or past cases.
    
    Current Insights:
    ${currentInsights}
    
    Knowledge Base:
    ${KNOWLEDGE_BASE_SOPS}
    
    Output:
    A set of citations and excerpts that explain or solve the issues found in the insights.
    If an insight matches a "Historical Case", explicitly mention it.
  `;

  const response = await ai.models.generateContent({
    model: 'gemini-2.5-flash',
    contents: prompt,
  });

  return response.text || "No relevant knowledge found.";
};

// 4. ReportAgent
export const generateFinalReport = async (
  summary: DataSummary,
  insights: string,
  knowledge: string
): Promise<AnalysisReport> => {
  const ai = getAiClient();

  // Create a minimal metric view for the report agent
  const keyMetrics = {
    rows: summary.n_rows,
    anomalies: summary.anomalies.total_flagged_rows,
    correlations: summary.top_correlations.slice(0, 3).map(c => `${c.pair.join(' vs ')} (${c.pearson.toFixed(2)})`)
  };

  const prompt = `
    You are the ReportAgent. Synthesize the final deliverables.
    
    Inputs:
    1. Key Metrics: ${JSON.stringify(keyMetrics)}
    2. Expert Insights: ${insights}
    3. Knowledge Context: ${knowledge}
    
    Task: Create a JSON object with:
    - technicalReport: Detailed Markdown for engineers. Sections: Analysis Methodology, Key Findings (Trend/Anomaly), Root Cause Hypothesis, Recommended Actions (citing SOPs).
    - executiveSummary: Concise Markdown for Plant Manager. Focus on: Yield Impact, Quality Risk, Business Decision. (Bullet points).
    - anomalies: A list of short strings describing top detected issues (e.g. "Temp drift > 5%").
  `;

  const response = await ai.models.generateContent({
    model: 'gemini-2.5-flash',
    contents: prompt,
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          technicalReport: { type: Type.STRING },
          executiveSummary: { type: Type.STRING },
          anomalies: { type: Type.ARRAY, items: { type: Type.STRING } },
        },
      },
    },
  });

  const text = response.text;
  if (!text) throw new Error("Failed to generate report");
  
  return JSON.parse(text) as AnalysisReport;
};