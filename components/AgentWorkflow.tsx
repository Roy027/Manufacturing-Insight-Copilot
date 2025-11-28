import React from 'react';
import { AgentState, AgentStatus } from '../types';
import { Bot, Database, Brain, FileText, CheckCircle2, Loader2, Circle, AlertCircle } from 'lucide-react';

interface AgentWorkflowProps {
  agents: AgentState[];
}

const AgentIcon = ({ role, className }: { role: string; className?: string }) => {
  if (role.includes('Analyst')) return <Database className={className} />;
  if (role.includes('Expert')) return <Brain className={className} />;
  if (role.includes('Librarian')) return <Bot className={className} />;
  return <FileText className={className} />;
};

const StatusIcon = ({ status }: { status: AgentStatus }) => {
  switch (status) {
    case AgentStatus.COMPLETED:
      return <CheckCircle2 className="w-5 h-5 text-green-500" />;
    case AgentStatus.WORKING:
      return <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />;
    case AgentStatus.ERROR:
      return <AlertCircle className="w-5 h-5 text-red-500" />;
    default:
      return <Circle className="w-5 h-5 text-slate-300" />;
  }
};

export const AgentWorkflow: React.FC<AgentWorkflowProps> = ({ agents }) => {
  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
      <h3 className="text-lg font-semibold text-slate-800 mb-6">Agent Orchestration</h3>
      <div className="relative">
        {/* Connector Line */}
        <div className="absolute left-6 top-6 bottom-6 w-0.5 bg-slate-100 -z-10" />
        
        <div className="space-y-8">
          {agents.map((agent, index) => (
            <div key={agent.id} className={`flex items-start gap-4 transition-all duration-300 ${agent.status === AgentStatus.IDLE ? 'opacity-50' : 'opacity-100'}`}>
              <div className={`p-3 rounded-lg border flex-shrink-0 ${
                agent.status === AgentStatus.WORKING 
                  ? 'bg-blue-50 border-blue-200 text-blue-600 ring-2 ring-blue-100' 
                  : agent.status === AgentStatus.COMPLETED
                  ? 'bg-green-50 border-green-200 text-green-600'
                  : 'bg-slate-50 border-slate-200 text-slate-500'
              }`}>
                <AgentIcon role={agent.role} className="w-6 h-6" />
              </div>
              
              <div className="flex-1 pt-1">
                <div className="flex justify-between items-center mb-1">
                  <h4 className="font-semibold text-slate-900">{agent.name}</h4>
                  <StatusIcon status={agent.status} />
                </div>
                <p className="text-xs font-medium uppercase tracking-wider text-slate-500 mb-2">{agent.role}</p>
                <div className={`text-sm rounded-md p-3 ${
                  agent.status === AgentStatus.WORKING ? 'bg-blue-50 text-blue-800' : 'bg-slate-50 text-slate-600'
                }`}>
                  {agent.message}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};