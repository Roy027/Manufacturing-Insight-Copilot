import React, { useState, useEffect } from 'react';
import * as XLSX from 'xlsx';
import { INITIAL_AGENTS, SAMPLE_CSV } from './constants';
import { AgentState, AgentStatus, DataSummary, AnalysisReport } from './types';
import { generateInsights, retrieveKnowledge, generateFinalReport } from './services/geminiService';
import { analyzeDataset } from './utils/dataProcessing';
import { AgentWorkflow } from './components/AgentWorkflow';
import { DataVisualization } from './components/DataVisualization';
import { ReportView } from './components/ReportView';
import { UploadCloud, Play, RotateCcw, Lock, BarChart3, Factory, FileText, FileSpreadsheet } from 'lucide-react';

export default function App() {
  const [hasKey, setHasKey] = useState(false);
  const [agents, setAgents] = useState<AgentState[]>(INITIAL_AGENTS);
  const [data, setData] = useState<DataSummary | null>(null);
  const [report, setReport] = useState<AnalysisReport | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  // --- API Key Management ---
  useEffect(() => {
    const checkKey = async () => {
      if ((window as any).aistudio) {
        const selected = await (window as any).aistudio.hasSelectedApiKey();
        setHasKey(selected);
      } else if (process.env.API_KEY) {
        setHasKey(true);
      }
    };
    checkKey();
  }, []);

  const handleSelectKey = async () => {
    if ((window as any).aistudio) {
      await (window as any).aistudio.openSelectKey();
      const selected = await (window as any).aistudio.hasSelectedApiKey();
      setHasKey(selected);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      let rawJson: any[] = [];
      
      if (file.name.endsWith('.csv')) {
        const text = await file.text();
        const workbook = XLSX.read(text, { type: 'string' });
        const worksheet = workbook.Sheets[workbook.SheetNames[0]];
        rawJson = XLSX.utils.sheet_to_json(worksheet);
      } else if (file.name.match(/\.xlsx?$/)) {
        const arrayBuffer = await file.arrayBuffer();
        const workbook = XLSX.read(arrayBuffer);
        const worksheet = workbook.Sheets[workbook.SheetNames[0]];
        rawJson = XLSX.utils.sheet_to_json(worksheet);
      }

      const summary = analyzeDataset(rawJson, file.name);
      setData(summary);
      resetAgents("Data loaded and profiled (Phases A-H).");

    } catch (err) {
      console.error("Error processing file:", err);
      alert("Error processing file. Please check the format.");
    }
  };

  const loadSampleData = () => {
    const workbook = XLSX.read(SAMPLE_CSV, { type: 'string' });
    const json = XLSX.utils.sheet_to_json(workbook.Sheets[workbook.SheetNames[0]]);
    const summary = analyzeDataset(json, "synthetic_production_data.csv");
    setData(summary);
    resetAgents("Sample data profiled.");
  };

  const resetAgents = (msg: string) => {
    setReport(null);
    setAgents(prev => prev.map(a => ({ 
      ...a, 
      status: AgentStatus.IDLE, 
      message: a.id === 'data-agent' ? msg : 'Waiting...' 
    })));
  };

  const updateAgent = (id: string, status: AgentStatus, message: string) => {
    setAgents(prev => prev.map(a => a.id === id ? { ...a, status, message } : a));
  };

  // --- Orchestrator ---
  const runAnalysis = async () => {
    if (!data || !hasKey) return;
    setIsProcessing(true);

    try {
      // 1. Data Agent (Handled by Client-Side Code Execution Tool)
      updateAgent('data-agent', AgentStatus.WORKING, 'Running Phase A-H statistical profiling...');
      await new Promise(r => setTimeout(r, 800)); 
      updateAgent('data-agent', AgentStatus.COMPLETED, `Profiled ${data.n_rows} rows, ${data.top_correlations.length} correlations.`);

      // 2. Insight Agent
      updateAgent('insight-agent', AgentStatus.WORKING, 'Analyzing profiles & trends (Gemini)...');
      const insights = await generateInsights(data);
      updateAgent('insight-agent', AgentStatus.COMPLETED, 'Technical insights generated.');

      // 3. Knowledge Agent
      updateAgent('knowledge-agent', AgentStatus.WORKING, 'Retrieving SOPs & Historical context...');
      const knowledge = await retrieveKnowledge(insights);
      updateAgent('knowledge-agent', AgentStatus.COMPLETED, 'Relevant knowledge retrieved.');

      // 4. Report Agent
      updateAgent('report-agent', AgentStatus.WORKING, 'Synthesizing technical & executive reports...');
      const finalReport = await generateFinalReport(data, insights, knowledge);
      setReport(finalReport);
      updateAgent('report-agent', AgentStatus.COMPLETED, 'Report generated.');

    } catch (error) {
      console.error(error);
      setAgents(prev => prev.map(a => a.status === AgentStatus.WORKING ? { ...a, status: AgentStatus.ERROR, message: 'Process failed.' } : a));
    } finally {
      setIsProcessing(false);
    }
  };

  if (!hasKey) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-50 p-4">
        <div className="max-w-md w-full bg-white rounded-2xl shadow-xl p-8 text-center border border-slate-100">
          <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-6">
            <Lock className="w-8 h-8 text-blue-600" />
          </div>
          <h1 className="text-2xl font-bold text-slate-900 mb-2">Authentication Required</h1>
          <p className="text-slate-600 mb-8">
            Please select a paid Google Cloud Project API Key to access the Manufacturing Insight Copilot.
          </p>
          <button
            onClick={handleSelectKey}
            className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-6 rounded-lg transition-colors flex items-center justify-center gap-2"
          >
            Select API Key
          </button>
          <div className="mt-6 text-xs text-slate-400">
            <a href="https://ai.google.dev/gemini-api/docs/billing" target="_blank" rel="noreferrer" className="underline hover:text-blue-500">
              Billing Information
            </a>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="bg-blue-600 p-2 rounded-lg">
              <Factory className="w-5 h-5 text-white" />
            </div>
            <h1 className="text-xl font-bold tracking-tight text-slate-900">
              Manufacturing Insight <span className="text-blue-600">Copilot</span>
            </h1>
          </div>
          <div className="flex items-center gap-4">
             <div className="text-xs px-3 py-1 bg-green-100 text-green-700 rounded-full font-medium flex items-center gap-1">
               <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
               System Online
             </div>
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 h-full">
          
          {/* Left Sidebar: Controls & Workflow */}
          <div className="lg:col-span-4 space-y-6">
            
            {/* Input Section */}
            <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
              <h2 className="text-lg font-semibold mb-4">Data Source</h2>
              
              {!data ? (
                <div className="space-y-4">
                  <div className="border-2 border-dashed border-slate-300 rounded-lg p-8 text-center hover:bg-slate-50 transition-colors relative">
                    <input 
                      type="file" 
                      accept=".csv, .xlsx, .xls"
                      onChange={handleFileUpload}
                      className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                    />
                    <UploadCloud className="w-10 h-10 text-slate-400 mx-auto mb-3" />
                    <p className="text-sm font-medium text-slate-700">Upload Data File</p>
                    <p className="text-xs text-slate-500 mt-1">Supported: CSV, Excel (.xlsx)</p>
                  </div>
                  
                  <div className="relative">
                    <div className="absolute inset-0 flex items-center">
                      <span className="w-full border-t border-slate-200" />
                    </div>
                    <div className="relative flex justify-center text-xs uppercase">
                      <span className="bg-white px-2 text-slate-500">Or</span>
                    </div>
                  </div>

                  <button 
                    onClick={loadSampleData}
                    className="w-full py-2 px-4 bg-white border border-slate-300 rounded-lg text-sm font-medium text-slate-700 hover:bg-slate-50 hover:text-slate-900 transition-colors"
                  >
                    Load Sample Batch Data
                  </button>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="flex items-center gap-3 p-3 bg-blue-50 rounded-lg border border-blue-100">
                    <FileSpreadsheet className="w-5 h-5 text-blue-600" />
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-blue-900 truncate">{data.fileName}</p>
                      <p className="text-xs text-blue-700">{data.n_rows} rows • {data.n_cols} cols</p>
                    </div>
                    <button 
                      onClick={() => { setData(null); resetAgents("Waiting for data..."); }}
                      className="p-1 hover:bg-blue-200 rounded text-blue-600"
                    >
                      <RotateCcw className="w-4 h-4" />
                    </button>
                  </div>

                  <button
                    onClick={runAnalysis}
                    disabled={isProcessing || agents[3].status === AgentStatus.COMPLETED}
                    className={`w-full py-3 px-4 rounded-lg flex items-center justify-center gap-2 font-semibold text-white transition-all shadow-md ${
                      isProcessing || agents[3].status === AgentStatus.COMPLETED
                        ? 'bg-slate-400 cursor-not-allowed'
                        : 'bg-blue-600 hover:bg-blue-700 hover:shadow-lg'
                    }`}
                  >
                    {isProcessing ? (
                      <>Processing...</>
                    ) : (
                      <>
                        <Play className="w-4 h-4 fill-current" /> Run Multi-Agent Analysis
                      </>
                    )}
                  </button>
                </div>
              )}
            </div>

            {/* Agent Status */}
            <AgentWorkflow agents={agents} />
          </div>

          {/* Right Content: Visualization & Reports */}
          <div className="lg:col-span-8 flex flex-col gap-6">
            {/* Visualizer (Always show if data exists) */}
            {data && (
              <div className="min-h-[500px]">
                <DataVisualization data={data} />
              </div>
            )}

            {/* Report Area */}
            {report ? (
              <div className="h-[600px]">
                <ReportView report={report} />
              </div>
            ) : (
              <div className="flex-1 bg-white rounded-xl shadow-sm border border-slate-200 border-dashed p-12 flex flex-col items-center justify-center text-center">
                <div className="w-16 h-16 bg-slate-50 rounded-full flex items-center justify-center mb-4">
                  <FileText className="w-8 h-8 text-slate-300" />
                </div>
                <h3 className="text-slate-900 font-medium mb-1">No Report Generated</h3>
                <p className="text-slate-500 text-sm max-w-sm">
                  Upload data and run the agent analysis to generate insights, anomaly detection, and reports.
                </p>
              </div>
            )}
          </div>

        </div>
      </main>
    </div>
  );
}
