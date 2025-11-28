import React, { useState } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, ScatterChart, Scatter
} from 'recharts';
import { DataSummary } from '../types';
import { TrendingUp, BarChart2, ScatterChart as ScatterIcon, Table as TableIcon, Settings2, AlertTriangle } from 'lucide-react';

interface DataVisualizationProps {
  data: DataSummary;
}

type ViewMode = 'trends' | 'distribution' | 'correlation' | 'stats';

export const DataVisualization: React.FC<DataVisualizationProps> = ({ data }) => {
  const [viewMode, setViewMode] = useState<ViewMode>('trends');
  
  const numericCols = Object.keys(data.numeric_profile);
  
  // State for Trends
  const [selectedTrendMetrics, setSelectedTrendMetrics] = useState<string[]>(
    numericCols.slice(0, 3)
  );

  // State for Distribution
  const [distributionMetric, setDistributionMetric] = useState<string>(
    numericCols[0] || ''
  );

  // State for Correlation
  const topCorr = data.top_correlations[0]?.pair || [];
  const [correlationX, setCorrelationX] = useState<string>(topCorr[0] || numericCols[0] || '');
  const [correlationY, setCorrelationY] = useState<string>(topCorr[1] || numericCols[1] || numericCols[0] || '');

  // Use raw data subset for scatter/line if available, otherwise we can't render points
  const plotData = data.raw_data_subset || [];
  const indexKey = data.time_profiles?.index_column || 'id';
  
  const colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899'];

  // Histogram Data from Profile
  const histogramData = data.numeric_profile[distributionMetric]?.histogram.map(b => ({
    name: b.range,
    count: b.count
  })) || [];

  const toggleTrendMetric = (metric: string) => {
    if (selectedTrendMetrics.includes(metric)) {
      setSelectedTrendMetrics(prev => prev.filter(m => m !== metric));
    } else {
      setSelectedTrendMetrics(prev => [...prev, metric]);
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 flex flex-col h-full overflow-hidden">
      
      {/* Header / Tabs */}
      <div className="border-b border-slate-100 p-2 flex gap-2 bg-slate-50/50 flex-wrap">
        <button
          onClick={() => setViewMode('trends')}
          className={`flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
            viewMode === 'trends' ? 'bg-blue-100 text-blue-700' : 'text-slate-600 hover:bg-slate-100'
          }`}
        >
          <TrendingUp className="w-4 h-4" /> Trends
        </button>
        <button
          onClick={() => setViewMode('distribution')}
          className={`flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
            viewMode === 'distribution' ? 'bg-blue-100 text-blue-700' : 'text-slate-600 hover:bg-slate-100'
          }`}
        >
          <BarChart2 className="w-4 h-4" /> Distribution
        </button>
        <button
          onClick={() => setViewMode('correlation')}
          className={`flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
            viewMode === 'correlation' ? 'bg-blue-100 text-blue-700' : 'text-slate-600 hover:bg-slate-100'
          }`}
        >
          <ScatterIcon className="w-4 h-4" /> Correlations
        </button>
        <button
          onClick={() => setViewMode('stats')}
          className={`flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
            viewMode === 'stats' ? 'bg-blue-100 text-blue-700' : 'text-slate-600 hover:bg-slate-100'
          }`}
        >
          <TableIcon className="w-4 h-4" /> Profiling
        </button>
      </div>

      <div className="flex-1 p-6 flex flex-col md:flex-row gap-6 overflow-hidden">
        
        {/* Controls Sidebar */}
        <div className="w-full md:w-64 flex-shrink-0 space-y-6 md:border-r border-slate-100 md:pr-6 overflow-y-auto max-h-[200px] md:max-h-full">
          
          {viewMode === 'trends' && (
            <div>
              <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3 flex items-center gap-2">
                <Settings2 className="w-3 h-3" /> Select Metrics
              </h4>
              <div className="space-y-2">
                {numericCols.map((col) => (
                  <label key={col} className="flex items-center gap-2 text-sm text-slate-700 cursor-pointer hover:bg-slate-50 p-1.5 rounded">
                    <input
                      type="checkbox"
                      checked={selectedTrendMetrics.includes(col)}
                      onChange={() => toggleTrendMetric(col)}
                      className="rounded border-slate-300 text-blue-600 focus:ring-blue-500"
                    />
                    {col}
                  </label>
                ))}
              </div>
              <div className="mt-4 p-3 bg-blue-50 rounded-lg">
                <p className="text-xs text-blue-700 font-medium">Index Column: {data.time_profiles?.index_column}</p>
                <p className="text-xs text-blue-600 mt-1">Range: {data.time_profiles?.index_range.min} - {data.time_profiles?.index_range.max}</p>
              </div>
            </div>
          )}

          {viewMode === 'distribution' && (
            <div>
              <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3">Metric to Analyze</h4>
              <select 
                value={distributionMetric} 
                onChange={(e) => setDistributionMetric(e.target.value)}
                className="w-full rounded-md border-slate-200 text-sm shadow-sm focus:border-blue-500 focus:ring-blue-500"
              >
                {numericCols.map(col => (
                  <option key={col} value={col}>{col}</option>
                ))}
              </select>
              <div className="mt-4 text-xs text-slate-500 space-y-1">
                 <p>Missing: {data.numeric_profile[distributionMetric]?.missing.count}</p>
                 <p>Zeros: {data.numeric_profile[distributionMetric]?.zeros.count}</p>
                 <p>Skew: {data.numeric_profile[distributionMetric]?.stats.skew.toFixed(2)}</p>
              </div>
            </div>
          )}

          {viewMode === 'correlation' && (
            <div className="space-y-4">
               <div className="p-3 bg-slate-50 rounded-lg">
                <h5 className="text-xs font-bold text-slate-600 mb-2">Top Detected Correlations</h5>
                <ul className="space-y-1">
                  {data.top_correlations.slice(0, 5).map((c, i) => (
                    <li key={i} className="text-xs flex justify-between cursor-pointer hover:text-blue-600"
                        onClick={() => { setCorrelationX(c.pair[0]); setCorrelationY(c.pair[1]); }}>
                      <span>{c.pair[0]} vs {c.pair[1]}</span>
                      <span className="font-mono font-medium">{c.pearson.toFixed(2)}</span>
                    </li>
                  ))}
                </ul>
              </div>
              <div>
                <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">X-Axis</h4>
                <select 
                  value={correlationX} 
                  onChange={(e) => setCorrelationX(e.target.value)}
                  className="w-full rounded-md border-slate-200 text-sm shadow-sm focus:border-blue-500 focus:ring-blue-500"
                >
                  {numericCols.map(col => (
                    <option key={col} value={col}>{col}</option>
                  ))}
                </select>
              </div>
              <div>
                <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">Y-Axis</h4>
                <select 
                  value={correlationY} 
                  onChange={(e) => setCorrelationY(e.target.value)}
                  className="w-full rounded-md border-slate-200 text-sm shadow-sm focus:border-blue-500 focus:ring-blue-500"
                >
                  {numericCols.map(col => (
                    <option key={col} value={col}>{col}</option>
                  ))}
                </select>
              </div>
            </div>
          )}
          
          {viewMode === 'stats' && (
             <div className="text-sm text-slate-500 space-y-4">
               <div className="p-4 bg-red-50 rounded-lg border border-red-100">
                  <h5 className="flex items-center gap-2 font-bold text-red-800 mb-2"><AlertTriangle className="w-4 h-4"/> Global Anomalies</h5>
                  <p className="text-red-700">Flagged Rows: {data.anomalies.total_flagged_rows}</p>
                  <p className="text-red-700">Rules: {data.anomalies.rules.map(r => r.rule).join(', ')}</p>
               </div>
               <p>Quick summary based on {data.n_rows} processed rows.</p>
             </div>
          )}

        </div>

        {/* Chart Area */}
        <div className="flex-1 min-h-[400px] w-full relative">
          
          {viewMode === 'trends' && (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={plotData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey={indexKey} stroke="#94a3b8" fontSize={11} tickLine={false} axisLine={false} />
                <YAxis stroke="#94a3b8" fontSize={11} tickLine={false} axisLine={false} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#fff', borderRadius: '8px', border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)' }}
                />
                <Legend />
                {selectedTrendMetrics.map((key, index) => (
                  <Line 
                    key={key} 
                    type="monotone" 
                    dataKey={key} 
                    stroke={colors[index % colors.length]} 
                    strokeWidth={2}
                    dot={false}
                    activeDot={{ r: 6 }}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          )}

          {viewMode === 'distribution' && (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={histogramData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                 <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                 <XAxis dataKey="name" stroke="#94a3b8" fontSize={11} tickLine={false} axisLine={false} />
                 <YAxis stroke="#94a3b8" fontSize={11} tickLine={false} axisLine={false} />
                 <Tooltip cursor={{fill: '#f8fafc'}} contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)' }} />
                 <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} name="Frequency" />
              </BarChart>
            </ResponsiveContainer>
          )}

          {viewMode === 'correlation' && (
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis type="number" dataKey={correlationX} name={correlationX} stroke="#94a3b8" fontSize={11} tickLine={false} axisLine={false} label={{ value: correlationX, position: 'insideBottom', offset: -10, fill: '#64748b', fontSize: 12 }} />
                <YAxis type="number" dataKey={correlationY} name={correlationY} stroke="#94a3b8" fontSize={11} tickLine={false} axisLine={false} label={{ value: correlationY, angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 12 }} />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)' }} />
                <Scatter name="Correlation" data={plotData} fill="#8b5cf6" />
              </ScatterChart>
            </ResponsiveContainer>
          )}

          {viewMode === 'stats' && (
            <div className="h-full overflow-y-auto pr-2">
              <table className="w-full text-sm text-left text-slate-600">
                <thead className="text-xs text-slate-500 uppercase bg-slate-50 sticky top-0">
                  <tr>
                    <th className="px-6 py-3 rounded-tl-lg">Metric</th>
                    <th className="px-6 py-3">Min</th>
                    <th className="px-6 py-3">Mean</th>
                    <th className="px-6 py-3">Max</th>
                    <th className="px-6 py-3">Outliers</th>
                    <th className="px-6 py-3 rounded-tr-lg">Trend</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {numericCols.map((col) => {
                    const stats = data.numeric_profile[col].stats;
                    const outliers = data.numeric_profile[col].outliers;
                    const trend = data.time_profiles?.metrics[col]?.trend;
                    return (
                      <tr key={col} className="hover:bg-slate-50 transition-colors">
                        <td className="px-6 py-4 font-medium text-slate-900">{col}</td>
                        <td className="px-6 py-4">{stats.min.toFixed(2)}</td>
                        <td className="px-6 py-4">{stats.mean.toFixed(2)}</td>
                        <td className="px-6 py-4">{stats.max.toFixed(2)}</td>
                        <td className="px-6 py-4">{outliers.count} ({ (outliers.fraction * 100).toFixed(1) }%)</td>
                        <td className="px-6 py-4">
                           {trend?.direction === 'up' && <span className="text-red-500 flex items-center gap-1">↗ {(trend.slope * 100).toFixed(4)}</span>}
                           {trend?.direction === 'down' && <span className="text-green-500 flex items-center gap-1">↘ {(trend.slope * 100).toFixed(4)}</span>}
                           {trend?.direction === 'flat' && <span className="text-slate-400">→</span>}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}

        </div>
      </div>
    </div>
  );
};
