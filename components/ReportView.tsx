import React from 'react';
import ReactMarkdown from 'react-markdown';
import { AnalysisReport } from '../types';
import { FileText, Briefcase } from 'lucide-react';

interface ReportViewProps {
  report: AnalysisReport;
}

export const ReportView: React.FC<ReportViewProps> = ({ report }) => {
  const [activeTab, setActiveTab] = React.useState<'technical' | 'executive'>('technical');

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden flex flex-col h-full">
      <div className="flex border-b border-slate-200">
        <button
          onClick={() => setActiveTab('technical')}
          className={`flex-1 py-4 text-sm font-medium flex items-center justify-center gap-2 transition-colors ${
            activeTab === 'technical' 
              ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50/50' 
              : 'text-slate-600 hover:text-slate-800 hover:bg-slate-50'
          }`}
        >
          <FileText className="w-4 h-4" />
          Technical Report
        </button>
        <button
          onClick={() => setActiveTab('executive')}
          className={`flex-1 py-4 text-sm font-medium flex items-center justify-center gap-2 transition-colors ${
            activeTab === 'executive' 
              ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50/50' 
              : 'text-slate-600 hover:text-slate-800 hover:bg-slate-50'
          }`}
        >
          <Briefcase className="w-4 h-4" />
          Executive Summary
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-6 bg-slate-50/30">
        <article className="prose prose-slate prose-sm max-w-none">
          <ReactMarkdown>
            {activeTab === 'technical' ? report.technicalReport : report.executiveSummary}
          </ReactMarkdown>
        </article>
      </div>
      
      {/* Anomalies Footer */}
      <div className="bg-red-50 border-t border-red-100 p-4">
        <h4 className="text-xs font-bold text-red-800 uppercase tracking-wider mb-2">Detected Anomalies</h4>
        <ul className="space-y-1">
          {report.anomalies.map((anomaly, idx) => (
            <li key={idx} className="text-sm text-red-700 flex items-start gap-2">
              <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-red-500 flex-shrink-0" />
              {anomaly}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};