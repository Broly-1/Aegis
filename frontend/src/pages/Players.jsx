import { useState, useEffect, useCallback } from 'react';
import { Search, Filter, ChevronLeft, ChevronRight, ShieldAlert, ShieldCheck, ArrowUpDown } from 'lucide-react';
import './Players.css';

const API = 'http://localhost:8000/api';

function formatCurrency(n) {
  if (n >= 1_000_000_000) return '$' + (n / 1_000_000_000).toFixed(1) + 'B';
  if (n >= 1_000_000) return '$' + (n / 1_000_000).toFixed(1) + 'M';
  if (n >= 1_000) return '$' + (n / 1_000).toFixed(1) + 'K';
  return '$' + (n?.toFixed?.(2) ?? n);
}

export default function Players() {
  const [players, setPlayers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [total, setTotal] = useState(0);
  const [search, setSearch] = useState('');
  const [filter, setFilter] = useState('all');
  const [sortBy, setSortBy] = useState('risk_score');
  const [sortOrder, setSortOrder] = useState('desc');
  const [searchInput, setSearchInput] = useState('');

  const fetchPlayers = useCallback(() => {
    setLoading(true);
    const params = new URLSearchParams({
      page, per_page: 50, search, filter, sort_by: sortBy, sort_order: sortOrder
    });
    fetch(`${API}/players?${params}`)
      .then(r => r.json())
      .then(d => {
        setPlayers(d.players || []);
        setTotalPages(d.total_pages || 1);
        setTotal(d.total || 0);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, [page, search, filter, sortBy, sortOrder]);

  useEffect(() => { fetchPlayers(); }, [fetchPlayers]);

  const handleSearch = (e) => {
    e.preventDefault();
    setSearch(searchInput);
    setPage(1);
  };

  const toggleSort = (field) => {
    if (sortBy === field) {
      setSortOrder(prev => prev === 'desc' ? 'asc' : 'desc');
    } else {
      setSortBy(field);
      setSortOrder('desc');
    }
    setPage(1);
  };

  const SortHeader = ({ field, children }) => (
    <th onClick={() => toggleSort(field)} className="sortable-th">
      <div className="th-content">
        {children}
        <ArrowUpDown size={12} className={sortBy === field ? 'sort-active' : ''} />
      </div>
    </th>
  );

  return (
    <div className="flex flex-col h-full bg-surface-dim overflow-hidden">
      {/* Header & Controls Container */}
      <div className="px-8 pt-8 pb-4 border-b border-white/5 bg-surface-dim/50 backdrop-blur-xl z-20">
        <div className="flex justify-between items-end mb-6">
          <div>
            <h1 className="font-h1 text-3xl font-bold text-on-surface">Player Intelligence</h1>
            <p className="font-body-main text-sm text-on-surface-variant mt-1">
              {total.toLocaleString()} entities indexed in the GNN knowledge graph.
            </p>
          </div>
          <div className="flex items-center gap-4">
            <span className="font-label-caps text-[10px] text-outline border border-outline-variant px-2 py-1 rounded bg-surface-container">SYSTEM_STABLE</span>
            <div className="flex h-2 w-2 rounded-full bg-tertiary shadow-[0_0_8px_#61f6b9]"></div>
          </div>
        </div>

        <div className="flex justify-between items-center gap-4">
          <form onSubmit={handleSearch} className="flex-1 max-w-xl group relative">
            <div className="absolute inset-y-0 left-3 flex items-center pointer-events-none text-outline group-focus-within:text-primary transition-colors">
              <Search size={16} />
            </div>
            <input
              type="text"
              placeholder="Search entity by ID or transaction hash..."
              value={searchInput}
              onChange={e => setSearchInput(e.target.value)}
              className="w-full bg-surface-container-lowest border border-outline-variant/30 rounded-lg pl-10 pr-4 py-2 text-sm text-on-surface placeholder-outline focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/20 transition-all"
            />
          </form>

          <div className="flex bg-surface-container-low p-1 rounded-lg border border-outline-variant/20">
            {['all', 'fraudulent', 'safe'].map(f => (
              <button
                key={f}
                className={`px-4 py-1.5 rounded-md text-xs font-semibold transition-all ${
                  filter === f 
                    ? 'bg-surface-container-highest text-primary shadow-sm' 
                    : 'text-outline hover:text-on-surface'
                }`}
                onClick={() => { setFilter(f); setPage(1); }}
              >
                {f.toUpperCase()}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Table Container */}
      <div className="flex-1 overflow-auto px-8 py-6">
        <div className="glass rounded-xl overflow-hidden border border-white/5 shadow-2xl">
          {loading ? (
            <div className="p-12 flex flex-col items-center justify-center gap-4">
              <div className="w-12 h-12 border-4 border-primary/20 border-t-primary rounded-full animate-spin"></div>
              <span className="font-data-mono text-xs text-outline">QUERYING_NEURAL_REGISTRY...</span>
            </div>
          ) : (
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-surface-container-low/50 border-b border-white/5">
                  <th className="py-4 px-6 font-label-caps text-[10px] text-outline uppercase tracking-wider">#</th>
                  <th className="py-4 px-6 font-label-caps text-[10px] text-outline uppercase tracking-wider">Entity ID</th>
                  <SortHeader field="risk_score">Risk Score</SortHeader>
                  <th className="py-4 px-6 font-label-caps text-[10px] text-outline uppercase tracking-wider">Prediction</th>
                  <th className="py-4 px-6 font-label-caps text-[10px] text-outline uppercase tracking-wider">Truth</th>
                  <SortHeader field="total_sent">Volume Out</SortHeader>
                  <SortHeader field="total_received">Volume In</SortHeader>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5 bg-surface-dim/30">
                {players.map((p, i) => (
                  <tr key={p.id} className="hover:bg-white/[0.02] transition-colors group">
                    <td className="py-4 px-6 font-data-mono text-xs text-outline">
                      {(page - 1) * 50 + i + 1}
                    </td>
                    <td className="py-4 px-6">
                      <div className="flex flex-col">
                        <span className="font-data-mono text-sm text-primary group-hover:text-primary-light transition-colors">
                          {p.id}
                        </span>
                        <span className="text-[10px] text-outline opacity-0 group-hover:opacity-100 transition-opacity">
                          MMORPG_ID_REF_X82
                        </span>
                      </div>
                    </td>
                    <td className="py-4 px-6">
                      <div className="flex items-center gap-3">
                        <div className="flex-1 bg-surface-container-highest h-1 rounded-full overflow-hidden min-w-[80px]">
                          <div 
                            className="h-full transition-all duration-1000"
                            style={{ 
                              width: `${p.risk_score * 100}%`,
                              background: p.risk_score > 0.8 ? '#ffb4ab' : 
                                         p.risk_score > 0.4 ? '#c0c1ff' : '#61f6b9',
                              boxShadow: `0 0 8px ${p.risk_score > 0.8 ? '#ffb4ab80' : '#61f6b980'}`
                            }}
                          />
                        </div>
                        <span className="font-data-mono text-xs text-on-surface w-10 text-right">
                          {(p.risk_score * 100).toFixed(0)}%
                        </span>
                      </div>
                    </td>
                    <td className="py-4 px-6">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold tracking-wider border ${
                        p.predicted_label === 'Fraudulent' 
                          ? 'bg-error/10 text-error border-error/20' 
                          : 'bg-tertiary/10 text-tertiary border-tertiary/20'
                      }`}>
                        {p.predicted_label.toUpperCase()}
                      </span>
                    </td>
                    <td className="py-4 px-6">
                      <span className={`text-[10px] font-data-mono ${
                        p.ground_truth === 'Fraudulent' ? 'text-error/70' : 'text-outline'
                      }`}>
                        {p.ground_truth}
                      </span>
                    </td>
                    <td className="py-4 px-6 font-data-mono text-xs text-on-surface">
                      {formatCurrency(p.total_sent)}
                    </td>
                    <td className="py-4 px-6 font-data-mono text-xs text-on-surface">
                      {formatCurrency(p.total_received)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        {/* Footer Pagination */}
        {!loading && (
          <div className="mt-6 flex justify-between items-center text-xs font-data-mono text-outline">
            <span>SHOWING_ENTITIES_OFFSET: {(page - 1) * 50} - {Math.min(page * 50, total)}</span>
            <div className="flex items-center gap-4">
              <button
                disabled={page <= 1}
                onClick={() => setPage(p => p - 1)}
                className="flex items-center gap-1 hover:text-primary disabled:opacity-30 disabled:hover:text-outline transition-colors"
              >
                <ChevronLeft size={14} /> PREV_NODE
              </button>
              <div className="px-3 py-1 bg-surface-container-low rounded border border-outline-variant/30 text-on-surface">
                PAGE_{page}
              </div>
              <button
                disabled={page >= totalPages}
                onClick={() => setPage(p => p + 1)}
                className="flex items-center gap-1 hover:text-primary disabled:opacity-30 disabled:hover:text-outline transition-colors"
              >
                NEXT_NODE <ChevronRight size={14} />
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
