import { 
  DataPoint, DataSummary, ColumnSchema, NumericProfile, 
  CategoricalProfile, CorrelationPair, TimeProfile, GlobalAnomalies,
  HistogramBin
} from '../types';

// --- CONFIGURATION ---
const MAX_CORR_PAIRS = 20;
const HIST_BINS = 10;
const MAX_CATEGORIES = 20;

// --- MATH HELPERS ---
const mean = (arr: number[]) => arr.reduce((a, b) => a + b, 0) / arr.length;

const variance = (arr: number[], m?: number) => {
  const avg = m ?? mean(arr);
  return mean(arr.map(x => Math.pow(x - avg, 2)));
};

const std = (arr: number[]) => Math.sqrt(variance(arr));

const percentile = (arr: number[], p: number) => {
  const sorted = [...arr].sort((a, b) => a - b);
  const pos = (sorted.length - 1) * p;
  const base = Math.floor(pos);
  const rest = pos - base;
  if ((sorted[base + 1] !== undefined)) {
    return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
  } else {
    return sorted[base];
  }
};

const skewness = (arr: number[], m: number, s: number) => {
  return mean(arr.map(x => Math.pow((x - m) / s, 3)));
};

const kurtosis = (arr: number[], m: number, s: number) => {
  return mean(arr.map(x => Math.pow((x - m) / s, 4))) - 3;
};

const pearsonCorrelation = (x: number[], y: number[]) => {
  const n = x.length;
  if (n !== y.length || n === 0) return 0;
  const mx = mean(x);
  const my = mean(y);
  const num = x.reduce((acc, xi, i) => acc + (xi - mx) * (y[i] - my), 0);
  const den = Math.sqrt(x.reduce((acc, xi) => acc + Math.pow(xi - mx, 2), 0) * y.reduce((acc, yi) => acc + Math.pow(yi - my, 2), 0));
  return den === 0 ? 0 : num / den;
};

const linearRegression = (y: number[], x: number[]) => {
  const n = y.length;
  const sumX = x.reduce((a, b) => a + b, 0);
  const sumY = y.reduce((a, b) => a + b, 0);
  const sumXY = x.reduce((a, xi, i) => a + xi * y[i], 0);
  const sumXX = x.reduce((a, xi) => a + xi * xi, 0);
  
  const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
  // Simple R2 approximation
  const yMean = sumY / n;
  const ssTot = y.reduce((a, yi) => a + Math.pow(yi - yMean, 2), 0);
  const ssRes = y.reduce((a, yi, i) => a + Math.pow(yi - (slope * x[i] + (sumY - slope * sumX) / n), 2), 0);
  const r2 = 1 - (ssRes / ssTot);
  
  return { slope, r2 };
};

// --- MAIN ANALYZER ---
export const analyzeDataset = (rawData: any[], fileName: string): DataSummary => {
  const n_rows = rawData.length;
  if (n_rows === 0) throw new Error("Empty dataset");

  // Normalize Data (remove empty keys, parse numbers)
  const data: DataPoint[] = rawData.map((row, idx) => {
    const newRow: DataPoint = { id: idx + 1 };
    Object.keys(row).forEach(key => {
      const cleanKey = key.trim();
      const val = row[key];
      const num = parseFloat(val);
      if (!isNaN(num) && typeof val !== 'boolean' && val !== '' && val !== null) {
        newRow[cleanKey] = num;
      } else {
        newRow[cleanKey] = val;
      }
    });
    return newRow;
  });

  const cols = Object.keys(data[0]).filter(k => k !== 'id');
  const n_cols = cols.length;

  // --- PHASE A: Load + Schema Profiling ---
  const schema: Record<string, ColumnSchema> = {};
  const numericCols: string[] = [];
  const categoricalCols: string[] = [];
  let indexCol = 'id'; // Default

  cols.forEach(col => {
    const val = data[0][col];
    let role: ColumnSchema['role'] = 'unknown';
    if (typeof val === 'number') {
      role = 'numeric';
      numericCols.push(col);
    } else {
      role = 'categorical';
      categoricalCols.push(col);
    }
    // Simple heuristic for Index/Time
    if (col.toLowerCase().includes('time') || col.toLowerCase().includes('date') || col.toLowerCase().includes('cycle')) {
      if (role === 'numeric') indexCol = col;
    }
    schema[col] = { dtype: typeof val, role };
  });

  // --- PHASE C: Numeric Statistics ---
  const numeric_profile: Record<string, NumericProfile> = {};
  
  numericCols.forEach(col => {
    const values = data.map(d => d[col] as number).filter(v => v !== undefined && v !== null && !isNaN(v));
    const n = values.length;
    if (n === 0) return;

    const m = mean(values);
    const s = std(values);
    const mn = Math.min(...values);
    const mx = Math.max(...values);
    
    // Outliers (IQR Method)
    const p25 = percentile(values, 0.25);
    const p75 = percentile(values, 0.75);
    const iqr = p75 - p25;
    const lower = p25 - 1.5 * iqr;
    const upper = p75 + 1.5 * iqr;
    const outlierCount = values.filter(v => v < lower || v > upper).length;

    // Histogram
    const binWidth = (mx - mn) / HIST_BINS;
    const histogram: HistogramBin[] = [];
    if (binWidth > 0) {
      for (let i = 0; i < HIST_BINS; i++) {
        const binMin = mn + i * binWidth;
        const binMax = mn + (i + 1) * binWidth;
        const count = values.filter(v => v >= binMin && (i === HIST_BINS - 1 ? v <= binMax : v < binMax)).length;
        histogram.push({ range: `${binMin.toFixed(1)} - ${binMax.toFixed(1)}`, count, min: binMin, max: binMax });
      }
    }

    numeric_profile[col] = {
      stats: {
        mean: m, std: s, min: mn, max: mx,
        p01: percentile(values, 0.01),
        p25, p50: percentile(values, 0.50), p75,
        p99: percentile(values, 0.99),
        skew: s !== 0 ? skewness(values, m, s) : 0,
        kurtosis: s !== 0 ? kurtosis(values, m, s) : 0,
      },
      histogram,
      missing: { count: n_rows - n, fraction: (n_rows - n) / n_rows },
      zeros: { count: values.filter(v => v === 0).length, fraction: values.filter(v => v === 0).length / n },
      outliers: { count: outlierCount, fraction: outlierCount / n }
    };
  });

  // --- PHASE D: Categorical Profiling ---
  const categorical_profile: Record<string, CategoricalProfile> = {};
  
  categoricalCols.forEach(col => {
    const values = data.map(d => String(d[col]));
    const counts: Record<string, number> = {};
    values.forEach(v => counts[v] = (counts[v] || 0) + 1);
    
    const uniqueVals = Object.keys(counts);
    const sortedVals = uniqueVals.sort((a, b) => counts[b] - counts[a]);
    const topK = sortedVals.slice(0, MAX_CATEGORIES).map(v => ({
      value: v,
      count: counts[v],
      fraction: counts[v] / n_rows
    }));
    
    const otherCount = sortedVals.slice(MAX_CATEGORIES).reduce((acc, v) => acc + counts[v], 0);

    categorical_profile[col] = {
      n_unique: uniqueVals.length,
      top_values: topK,
      other_fraction: otherCount / n_rows,
      missing: { count: 0, fraction: 0 } // simplified
    };
  });

  // --- PHASE E: Correlation Extraction ---
  const top_correlations: CorrelationPair[] = [];
  if (numericCols.length > 1) {
    const pairs: { pair: [string, string], val: number }[] = [];
    for (let i = 0; i < numericCols.length; i++) {
      for (let j = i + 1; j < numericCols.length; j++) {
        const c1 = numericCols[i];
        const c2 = numericCols[j];
        const v1 = data.map(d => d[c1] as number);
        const v2 = data.map(d => d[c2] as number);
        const corr = Math.abs(pearsonCorrelation(v1, v2));
        if (!isNaN(corr)) {
           pairs.push({ pair: [c1, c2], val: corr });
        }
      }
    }
    // Sort by strength
    pairs.sort((a, b) => b.val - a.val);
    top_correlations.push(...pairs.slice(0, MAX_CORR_PAIRS).map(p => ({
      pair: p.pair,
      pearson: p.val,
      n_samples: n_rows
    })));
  }

  // --- PHASE F: Time / Index Profiles ---
  const time_profiles: TimeProfile = {
    index_column: indexCol,
    index_range: { min: data[0][indexCol], max: data[n_rows - 1][indexCol] },
    metrics: {}
  };

  const indexValues = data.map((d, i) => typeof d[indexCol] === 'number' ? d[indexCol] as number : i);
  
  numericCols.forEach(col => {
    if (col === indexCol) return;
    const values = data.map(d => d[col] as number);
    const { slope, r2 } = linearRegression(values, indexValues);
    
    // Simple change point (max delta in rolling mean)
    // Simplified: Find index where diff from global mean is max (proxy)
    let maxDiff = 0;
    let cpIndex = 0;
    const globalMean = mean(values);
    values.forEach((v, i) => {
      if (Math.abs(v - globalMean) > maxDiff) {
        maxDiff = Math.abs(v - globalMean);
        cpIndex = i;
      }
    });

    time_profiles.metrics[col] = {
      trend: { 
        slope, 
        r2, 
        direction: Math.abs(slope) < 0.001 ? 'flat' : slope > 0 ? 'up' : 'down' 
      },
      change_points: [{ index: cpIndex, value: values[cpIndex], delta_mean: maxDiff }]
    };
  });

  // --- PHASE G: Global Anomaly Indicators ---
  // Simple rule: count rows with > 1 outlier across columns
  let flaggedRows = 0;
  data.forEach(row => {
    let outlierCount = 0;
    numericCols.forEach(col => {
      const val = row[col] as number;
      const stats = numeric_profile[col]?.stats;
      if (stats) {
        // approximate check using computed mean/std from profile to save time
        if (Math.abs(val - stats.mean) > 3 * stats.std) outlierCount++;
      }
    });
    if (outlierCount > 0) flaggedRows++;
  });

  const anomalies: GlobalAnomalies = {
    total_flagged_rows: flaggedRows,
    fraction: flaggedRows / n_rows,
    rules: [{ rule: "Value > 3 Sigma", count: flaggedRows }]
  };

  // --- FINAL RETURN ---
  return {
    fileName,
    n_rows,
    n_cols,
    schema,
    numeric_profile,
    categorical_profile,
    top_correlations,
    time_profiles,
    anomalies,
    sample_rows: data.slice(0, 5),
    raw_data_subset: data // Keep full data for visualization UI only
  };
};