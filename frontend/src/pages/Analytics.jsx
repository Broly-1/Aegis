import { useState, useEffect } from 'react';
import {
  BarChart3, TrendingUp, AlertTriangle, PieChart as PieIcon
} from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, CartesianGrid, AreaChart, Area, Legend
} from 'recharts';
import './Analytics.css';

const API = import.meta.env.VITE_API_URL || 'https://aegis-api-762161152188.us-central1.run.app/api';

const COLORS = ['#6366f1', '#22d3ee', '#34d399', '#fbbf24', '#f87171', '#a78bfa', '#fb7185'];

function formatNumber(n) {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
  if (n >= 1_000) return (n / 1_000).toFixed(1) + 'K';
  return n?.toLocaleString?.() ?? n;
}

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="chart-tooltip">
        <p className="chart-tooltip-label">{label}</p>
        {payload.map((p, i) => (
          <p key={i} style={{ color: p.color }}>
            {p.name}: {typeof p.value === 'number' ? p.value.toLocaleString() : p.value}
          </p>
        ))}
      </div>
    );
  }
  return null;
};

export default function Analytics() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API}/dashboard`)
      .then(r => r.json())
      .then(d => { setData(d); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="page-container">
        <div className="loading-grid">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="skeleton" style={{ height: 350 }} />
          ))}
        </div>
      </div>
    );
  }

  if (!data) return null;

  // Prepare data
  const tradeTypeData = Object.entries(data.trade_types || {}).map(([name, value]) => ({
    name, all: value, fraud: data.fraud_by_trade_type?.[name] || 0
  }));

  const valueDistData = Object.entries(data.value_distribution || {}).map(([name, value]) => ({
    name, value
  }));

  const monthlyData = data.monthly_trends || [];

  // Classification report data
  const report = data.classification_report || {};
  const reportData = [
    {
      metric: 'Precision',
      safe: ((report.Safe?.precision || 0) * 100).toFixed(1),
      fraudulent: ((report.Fraudulent?.precision || 0) * 100).toFixed(1),
    },
    {
      metric: 'Recall',
      safe: ((report.Safe?.recall || 0) * 100).toFixed(1),
      fraudulent: ((report.Fraudulent?.recall || 0) * 100).toFixed(1),
    },
    {
      metric: 'F1-Score',
      safe: ((report.Safe?.['f1-score'] || 0) * 100).toFixed(1),
      fraudulent: ((report.Fraudulent?.['f1-score'] || 0) * 100).toFixed(1),
    },
  ];

  return (
    <div className="flex flex-col h-full bg-surface-dim overflow-y-auto">
      {/* Page Header */}
      <div className="px-8 py-8 border-b border-white/5 bg-surface-dim/50 backdrop-blur-xl">
        <h1 className="font-h1 text-3xl font-bold text-on-surface">Advanced Analytics</h1>
        <p className="font-body-main text-sm text-on-surface-variant mt-1">
          GNN Knowledge Graph Telemetry & Model Performance Breakdown
        </p>
      </div>

      <div className="p-8 space-y-8">
        {/* Top Row: Metrics & Distribution */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="glass-panel rounded-xl p-6 flex flex-col gap-6">
            <h3 className="font-h2 text-sm text-on-surface flex items-center gap-2">
              <BarChart3 size={18} className="text-primary" />
              Volume Analysis: All vs Fraudulent
            </h3>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={tradeTypeData} barGap={4}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                  <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={formatNumber} />
                  <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.05)' }} />
                  <Bar dataKey="all" name="All Trades" fill="rgba(34, 211, 238, 0.4)" stroke="#22d3ee" radius={[2, 2, 0, 0]} barSize={16} />
                  <Bar dataKey="fraud" name="Fraudulent" fill="rgba(248, 113, 113, 0.4)" stroke="#f87171" radius={[2, 2, 0, 0]} barSize={16} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="glass-panel rounded-xl p-6 flex flex-col gap-6">
            <h3 className="font-h2 text-sm text-on-surface flex items-center gap-2">
              <PieIcon size={18} className="text-secondary" />
              Transaction Value Distribution
            </h3>
            <div className="h-[300px] relative">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={valueDistData}
                    cx="50%" cy="50%"
                    innerRadius={60}
                    outerRadius={100}
                    paddingAngle={4}
                    dataKey="value"
                    stroke="none"
                  >
                    {valueDistData.map((_, i) => (
                      <Cell key={i} fill={COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip content={<CustomTooltip />} />
                  <Legend layout="vertical" align="right" verticalAlign="middle" wrapperStyle={{ fontSize: '10px' }} />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Middle Row: Trends */}
        <div className="glass-panel rounded-xl p-6 flex flex-col gap-6">
          <h3 className="font-h2 text-sm text-on-surface flex items-center gap-2">
            <TrendingUp size={18} className="text-tertiary" />
            Monthly Threat Landscape Trends
          </h3>
          <div className="h-[350px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={monthlyData}>
                <defs>
                  <linearGradient id="areaGlow" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#6366f1" stopOpacity={0.1} />
                    <stop offset="100%" stopColor="#6366f1" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                <XAxis dataKey="month" tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false} />
                <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false} tickFormatter={formatNumber} />
                <Tooltip content={<CustomTooltip />} />
                <Area 
                  type="monotone" 
                  dataKey="transactions" 
                  stroke="#6366f1" 
                  fill="url(#areaGlow)" 
                  strokeWidth={2}
                  dot={{ fill: '#6366f1', r: 4 }}
                  activeDot={{ r: 6, stroke: '#fff', strokeWidth: 2 }}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Bottom Row: Model Health */}
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
          <div className="xl:col-span-2 glass-panel rounded-xl p-6 flex flex-col gap-6">
            <h3 className="font-h2 text-sm text-on-surface flex items-center gap-2">
              <AlertTriangle size={18} className="text-error" />
              GNN Classification Accuracy (Safe vs Fraud)
            </h3>
            <div className="space-y-6">
              {reportData.map(r => (
                <div key={r.metric} className="flex flex-col gap-2">
                  <div className="flex justify-between items-center text-xs font-data-mono">
                    <span className="text-outline uppercase">{r.metric}</span>
                    <div className="flex gap-4">
                      <span className="text-tertiary">SAFE: {r.safe}%</span>
                      <span className="text-error">FRAUD: {r.fraudulent}%</span>
                    </div>
                  </div>
                  <div className="flex h-1.5 gap-1 rounded-full overflow-hidden bg-surface-container-highest">
                    <div className="bg-tertiary/40 h-full" style={{ width: `${r.safe}%` }}></div>
                    <div className="bg-error/40 h-full" style={{ width: `${r.fraudulent}%` }}></div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="glass-panel-elevated rounded-xl p-6 flex flex-col justify-between gap-4">
            <h3 className="font-h2 text-xs text-outline uppercase tracking-widest">Inference Summary</h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="glass-panel rounded-lg p-4">
                <div className="text-2xl font-bold text-primary tracking-tight">{data.summary.accuracy}%</div>
                <div className="text-[10px] text-outline uppercase mt-1">Accuracy</div>
              </div>
              <div className="glass-panel rounded-lg p-4">
                <div className="text-2xl font-bold text-tertiary tracking-tight">{data.summary.detection_rate}%</div>
                <div className="text-[10px] text-outline uppercase mt-1">Recall</div>
              </div>
              <div className="glass-panel rounded-lg p-4">
                <div className="text-2xl font-bold text-on-surface tracking-tight">{data.summary.total_ground_truth_fraud}</div>
                <div className="text-[10px] text-outline uppercase mt-1">Real Fraud</div>
              </div>
              <div className="glass-panel rounded-lg p-4">
                <div className="text-2xl font-bold text-on-surface tracking-tight">{formatNumber(data.summary.total_flagged)}</div>
                <div className="text-[10px] text-outline uppercase mt-1">Flagged</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
