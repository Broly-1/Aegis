import { useState, useEffect, useRef, useCallback } from 'react';
import { Network, Maximize2 } from 'lucide-react';
import ForceGraph2D from 'react-force-graph-2d';
import './GraphView.css';

const API = import.meta.env.VITE_API_URL || 'https://aegis-api-762161152188.us-central1.run.app/api';

export default function GraphView() {
  const [graphData, setGraphData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [windowSize, setWindowSize] = useState({ width: 800, height: 600 });
  const containerRef = useRef(null);
  const graphRef = useRef(null);

  useEffect(() => {
    fetch(`${API}/graph`)
      .then(r => r.json())
      .then(d => {
        const formattedData = {
          nodes: d.nodes,
          links: d.edges || d.links
        };
        setGraphData(formattedData);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  useEffect(() => {
    const handleResize = () => {
      if (containerRef.current) {
        setWindowSize({
          width: containerRef.current.clientWidth,
          height: containerRef.current.clientHeight
        });
      }
    };
    
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [loading]);

  const resetView = () => {
    if (graphRef.current) {
      graphRef.current.zoomToFit(400);
    }
  };

  const drawNode = useCallback((node, ctx, globalScale) => {
    const risk = node.risk_score || 0;
    const isHighRisk = risk > 0.8;
    const isMediumRisk = risk > 0.6;
    const isLowRisk = risk > 0.4;
    
    const color = isHighRisk ? '#f87171' : 
                  isMediumRisk ? '#fbbf24' : 
                  isLowRisk ? '#6366f1' : '#34d399';
    
    const size = isHighRisk ? 6 : 4;
    
    // Draw glow for high risk
    if (isHighRisk || isMediumRisk) {
      ctx.beginPath();
      ctx.arc(node.x, node.y, size + 4, 0, 2 * Math.PI, false);
      ctx.fillStyle = isHighRisk ? 'rgba(248, 113, 113, 0.2)' : 'rgba(251, 191, 36, 0.15)';
      ctx.fill();
    }
    
    ctx.beginPath();
    ctx.arc(node.x, node.y, size, 0, 2 * Math.PI, false);
    ctx.fillStyle = color;
    ctx.fill();
    
    // Optional label on high zoom
    if (globalScale > 2 && isHighRisk) {
      const label = node.id;
      const fontSize = 12/globalScale;
      ctx.font = `${fontSize}px Sans-Serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
      ctx.fillText(label, node.x, node.y + size + fontSize);
    }
  }, []);

  return (
    <div className="page-container graph-page">
      {/* Header */}
      <div className="page-header animate-fade-in">
        <div>
          <h1 className="page-title">Fraud Network Graph</h1>
          <p className="page-subtitle">
            Interactive visualization of fraudulent transaction connections
          </p>
        </div>
      </div>

      <div className="graph-container glass-panel animate-fade-in-delay-1" ref={containerRef}>
        {/* Controls */}
        <div className="graph-controls">
          <button className="graph-ctrl-btn" onClick={resetView} title="Reset View">
            <Maximize2 size={16} />
          </button>
        </div>

        {/* Legend */}
        <div className="graph-legend">
          <div className="legend-row">
            <span className="legend-circle" style={{ background: '#f87171' }} />
            <span>Critical Risk (&gt;80%)</span>
          </div>
          <div className="legend-row">
            <span className="legend-circle" style={{ background: '#fbbf24' }} />
            <span>High Risk (60-80%)</span>
          </div>
          <div className="legend-row">
            <span className="legend-circle" style={{ background: '#6366f1' }} />
            <span>Medium Risk (40-60%)</span>
          </div>
          <div className="legend-row">
            <span className="legend-circle" style={{ background: '#34d399' }} />
            <span>Low Risk (&lt;40%)</span>
          </div>
        </div>

        {/* Canvas */}
        {loading ? (
          <div className="graph-loading">
            <Network size={48} className="graph-loading-icon" />
            <p>Loading network structure...</p>
          </div>
        ) : (
          graphData && (
            <ForceGraph2D
              ref={graphRef}
              width={windowSize.width}
              height={windowSize.height}
              graphData={graphData}
              nodeCanvasObject={drawNode}
              nodeRelSize={4}
              linkColor={() => 'rgba(99, 102, 241, 0.2)'}
              linkWidth={0.5}
              backgroundColor="transparent"
              cooldownTicks={100}
              d3AlphaDecay={0.05}
              d3VelocityDecay={0.2}
              onEngineStop={() => {
                if (graphRef.current) {
                  graphRef.current.zoomToFit(400);
                }
              }}
              nodeLabel={(node) => `ID: ${node.id}<br/>Risk: ${(node.risk_score * 100).toFixed(1)}%`}
            />
          )
        )}

        {/* Stats */}
        {graphData && (
          <div className="graph-stats">
            <span>{graphData.nodes?.length || 0} nodes</span>
            <span className="graph-stats-divider">|</span>
            <span>{graphData.links?.length || 0} edges</span>
          </div>
        )}
      </div>
    </div>
  );
}
