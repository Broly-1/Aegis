import { useState, useEffect } from 'react';
import { ShieldAlert } from 'lucide-react';
import './Dashboard.css';

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

const PIE_COLORS = ['#6366f1', '#22d3ee', '#34d399', '#fbbf24', '#f87171'];

function formatNumber(n) {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
  if (n >= 1_000) return (n / 1_000).toFixed(1) + 'K';
  return n?.toLocaleString?.() ?? n;
}

function formatCurrency(n) {
  if (n >= 1_000_000_000_000) return '$' + (n / 1_000_000_000_000).toFixed(1) + 'T';
  if (n >= 1_000_000_000) return '$' + (n / 1_000_000_000).toFixed(1) + 'B';
  if (n >= 1_000_000) return '$' + (n / 1_000_000).toFixed(1) + 'M';
  return n == null ? '$0' : '$' + n.toLocaleString();
}

export default function Dashboard() {
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
          {[...Array(8)].map((_, i) => (
            <div key={i} className="skeleton" style={{ height: 130 }} />
          ))}
        </div>
      </div>
    );
  }

  if (!data || data.error) {
    return (
      <div className="page-container">
        <div className="error-state">
          <ShieldAlert size={48} />
          <h2>Data Not Available</h2>
          <p>Run <code>python inference.py</code> then start the API server.</p>
        </div>
      </div>
    );
  }

  const { summary } = data;

  const riskData = Object.entries(data.risk_distribution || {}).map(([name, value]) => ({
    name: name.split(' ')[0] + ' ' + (name.split(' ')[1] || ''),
    fullName: name,
    value,
  }));

  return (
<main className="flex-1 flex flex-col min-h-screen relative overflow-x-hidden p-8">

<div className="absolute top-0 left-1/4 w-96 h-96 bg-primary/5 rounded-full blur-[120px] pointer-events-none -z-10"></div>
<div className="absolute bottom-1/4 right-0 w-[500px] h-[500px] bg-secondary-container/5 rounded-full blur-[150px] pointer-events-none -z-10"></div>

<div className="p-margin max-w-7xl mx-auto w-full space-y-lg text-white">

<div className="flex justify-between items-end mb-xl border-b border-white/10 pb-sm">
<div>
<h1 className="font-h1 text-h1 text-on-surface">Dashboard Overview</h1>
<p className="font-body-main text-body-main text-on-surface-variant mt-1">Real-time GNN inference telemetry and threat landscape.</p>
</div>
<div className="hidden sm:flex items-center gap-md">
<span className="font-label-caps text-label-caps text-outline bg-surface-container py-1 px-2 rounded-DEFAULT border border-outline-variant">LIVE DATA STREAM</span>
<span className="relative flex h-3 w-3">
<span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-tertiary opacity-75"></span>
<span className="relative inline-flex rounded-full h-3 w-3 bg-tertiary"></span>
</span>
</div>
</div>

<div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-md">

<div className="glass-panel rounded-lg p-md relative overflow-hidden stat-card-accent stat-card-cyan hover:scale-[1.02] transition-transform duration-300">
<div className="flex justify-between items-start mb-4">
<span className="font-label-caps text-label-caps text-on-surface-variant uppercase">Total Players</span>
<span className="material-symbols-outlined text-outline">group</span>
</div>
<div className="font-data-lg text-h1 text-on-surface tracking-tight">{formatNumber(summary.total_players)}</div>
<div className="mt-2 text-xs text-tertiary flex items-center gap-1 font-body-sm">
<span className="material-symbols-outlined text-[14px]">trending_up</span> {formatNumber(summary.total_transactions)} transactions
</div>
</div>

<div className="glass-panel rounded-lg p-md relative overflow-hidden stat-card-accent stat-card-emerald hover:scale-[1.02] transition-transform duration-300">
<div className="flex justify-between items-start mb-4">
<span className="font-label-caps text-label-caps text-on-surface-variant uppercase">Detection Rate</span>
<span className="material-symbols-outlined text-outline">radar</span>
</div>
<div className="font-data-lg text-h1 text-tertiary glow-emerald tracking-tight">{summary.detection_rate}%</div>
<div className="mt-2 text-xs text-tertiary flex items-center gap-1 font-body-sm">
<span className="material-symbols-outlined text-[14px]">trending_up</span> {summary.total_ground_truth_fraud} cases
</div>
</div>

<div className="glass-panel rounded-lg p-md relative overflow-hidden stat-card-accent stat-card-rose hover:scale-[1.02] transition-transform duration-300">
<div className="flex justify-between items-start mb-4">
<span className="font-label-caps text-label-caps text-on-surface-variant uppercase">Active Threats</span>
<span className="material-symbols-outlined text-outline">warning</span>
</div>
<div className="font-data-lg text-h1 text-error glow-rose tracking-tight">{formatNumber(summary.total_flagged)}</div>
<div className="mt-2 text-xs text-error flex items-center gap-1 font-body-sm">
<span className="material-symbols-outlined text-[14px]">priority_high</span> {((summary.total_flagged / summary.total_players) * 100).toFixed(1)}% of players
</div>
</div>

<div className="glass-panel rounded-lg p-md relative overflow-hidden stat-card-accent stat-card-indigo hover:scale-[1.02] transition-transform duration-300">
<div className="flex justify-between items-start mb-4">
<span className="font-label-caps text-label-caps text-on-surface-variant uppercase">Safe Trades</span>
<span className="material-symbols-outlined text-outline">verified_user</span>
</div>
<div className="font-data-lg text-h1 text-secondary tracking-tight">{formatNumber(summary.total_safe)}</div>
<div className="mt-2 text-xs text-on-surface-variant flex items-center gap-1 font-body-sm">
<span className="material-symbols-outlined text-[14px]">check_circle</span> Verified as legitimate
</div>
</div>
</div>

<div className="grid grid-cols-1 lg:grid-cols-3 gap-md mt-6">

<div className="lg:col-span-2 glass-panel rounded-xl p-lg">
<h2 className="font-h2 text-h2 text-on-surface mb-6 flex items-center gap-2 border-b border-white/5 pb-2">
<span className="material-symbols-outlined text-primary">model_training</span>
                        Model Performance
                    </h2>
<div className="space-y-6">

<div>
<div className="flex justify-between mb-1">
<span className="font-label-caps text-label-caps text-on-surface-variant">ACCURACY</span>
<span className="font-data-mono text-data-mono text-primary">{summary.accuracy}%</span>
</div>
<div className="w-full bg-surface-container-high rounded-full h-1.5 overflow-hidden">
<div className="bg-primary h-1.5 rounded-full" style={{ width: `${summary.accuracy}%`, boxShadow: '0 0 8px rgba(34, 211, 238, 0.5)' }}></div>
</div>
</div>

<div>
<div className="flex justify-between mb-1">
<span className="font-label-caps text-label-caps text-on-surface-variant">PRECISION</span>
<span className="font-data-mono text-data-mono text-tertiary">{summary.precision}%</span>
</div>
<div className="w-full bg-surface-container-high rounded-full h-1.5 overflow-hidden">
<div className="bg-tertiary h-1.5 rounded-full" style={{ width: `${summary.precision}%`, boxShadow: '0 0 8px rgba(97, 246, 185, 0.5)' }}></div>
</div>
</div>

<div>
<div className="flex justify-between mb-1">
<span className="font-label-caps text-label-caps text-on-surface-variant">F1-SCORE</span>
<span className="font-data-mono text-data-mono text-on-surface">{summary.f1_score}%</span>
</div>
<div className="w-full bg-surface-container-high rounded-full h-1.5 overflow-hidden">
<div className="bg-on-surface h-1.5 rounded-full" style={{ width: `${summary.f1_score}%` }}></div>
</div>
</div>
</div>
</div>

<div className="glass-panel-elevated rounded-xl p-lg flex flex-col justify-between">
<h2 className="font-h2 text-h2 text-on-surface mb-4 flex items-center gap-2">
<span className="material-symbols-outlined text-secondary">donut_large</span>
                        Risk Distribution
                    </h2>
<div className="flex-1 flex items-center justify-center py-4">
<div className="relative w-40 h-40 rounded-full border-8 border-surface-container-high" style={{ background: 'conic-gradient(#61f6b9 0% 85%, #8aebff 85% 95%, #6366f1 95% 99%, #ffb4ab 99% 100%)' }}>
<div className="absolute inset-0 m-auto w-24 h-24 bg-surface-dim rounded-full flex items-center justify-center glass-panel">
<span className="font-data-lg text-data-lg text-on-surface">100%</span>
</div>
</div>
</div>
<div className="space-y-2 mt-4 font-body-sm text-body-sm">
{riskData.slice(0,4).map((item, i) => (
  <div key={item.name} className="flex justify-between items-center">
  <div className="flex items-center gap-2"><div className="w-2 h-2 rounded-full" style={{ background: PIE_COLORS[i] }}></div><span className="text-on-surface-variant">{item.name}</span></div>
  <span className="font-data-mono text-data-mono text-white">{formatNumber(item.value)}</span>
  </div>
))}
</div>
</div>
</div>

<div className="glass-panel rounded-xl overflow-hidden mt-6">
<div className="p-lg border-b border-white/5 flex justify-between items-center bg-surface-container-low/50 p-6">
<h2 className="font-h2 text-h2 text-on-surface flex items-center gap-2 text-white font-semibold">
<span className="material-symbols-outlined text-error">local_police</span>
                        Top Flagged Players
                    </h2>
</div>
<div className="overflow-x-auto">
<table className="w-full text-left border-collapse">
<thead>
<tr className="border-b border-white/5 bg-surface-container-lowest/30">
<th className="py-4 px-6 font-label-caps text-label-caps text-on-surface-variant uppercase">Player ID</th>
<th className="py-4 px-6 font-label-caps text-label-caps text-on-surface-variant uppercase">Risk Score</th>
<th className="py-4 px-6 font-label-caps text-label-caps text-on-surface-variant uppercase">Predicted Label</th>
<th className="py-4 px-6 font-label-caps text-label-caps text-on-surface-variant uppercase text-right">Total Sent</th>
</tr>
</thead>
<tbody className="font-body-sm text-body-sm divide-y divide-white/5 text-white">

{(data.top_flagged_players || []).slice(0, 5).map((p) => (
<tr key={p.id} className="hover:bg-white/5 transition-colors">
<td className="py-4 px-6 font-data-mono text-data-mono text-primary">{p.id}</td>
<td className="py-4 px-6 font-data-mono text-data-mono text-error glow-rose">{(p.risk_score * 100).toFixed(1)}%</td>
<td className="py-4 px-6">
<span className={`px-2 py-1 text-xs rounded-DEFAULT font-label-caps tracking-wider border ${p.predicted_label === 'Fraudulent' ? 'bg-error-container text-on-error-container border-error/30' : 'bg-secondary-container/50 text-secondary border-secondary/30'}`}>{p.predicted_label.toUpperCase()}</span>
</td>
<td className="py-4 px-6 font-data-mono text-data-mono text-right text-on-surface">{formatCurrency(p.total_sent)}</td>
</tr>
))}

</tbody>
</table>
</div>
</div>
</div>
</main>
  );
}
