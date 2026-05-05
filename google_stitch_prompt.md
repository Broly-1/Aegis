# Prompt for Google Stitch: GNN Fraud Detection Dashboard UI

**Context**: 
I have a React + Vite application that serves as a dashboard for a Graph Neural Network (GNN) Fraud Detection system for an MMORPG. The backend provides real-time stats and risk scores for 1.3 million players. I want to build an incredibly premium, dark-themed, glassmorphic UI.

**Task**: 
Please redesign and generate the React components and CSS for the following UI based on a unified, high-end "cyber security" design system. The components currently use `lucide-react` for icons and `recharts` for charts.

## Design System & Vibe
- **Theme**: Deep Dark Mode (e.g., `#0a0e1a` for background) with vibrant cyber accents (Neon Cyan `#22d3ee`, Indigo `#6366f1`, Emerald `#34d399`, and Rose `#fb7185`).
- **Aesthetics**: Heavy use of "glassmorphism" (frosted glass, backdrop blur, semi-transparent borders).
- **Animations**: Include subtle micro-animations (e.g., hover lifts, glowing borders on focus, staggered fade-ins for list items).
- **Typography**: Modern, technical fonts like `Inter` for standard text and `JetBrains Mono` for IDs, risk scores, and data points.

## Components to Redesign

### 1. `Sidebar.jsx`
- **Purpose**: Main navigation menu.
- **Content**: A brand header ("FraudGuard" with a shield icon), navigation links (`/` Dashboard, `/players` Players, `/graph` Network Graph, `/analytics` Analytics), and a footer showing the model version with a pulsing green "online" dot.
- **Requirement**: Make the active state glow and the hover state smooth.

### 2. `StatCard.jsx`
- **Purpose**: Reusable top-level metric card (e.g., "Total Players: 1.3M", "Detection Rate: 66%").
- **Content**: Needs an icon, a main label, a large value, and a smaller sub-value.
- **Requirement**: Add a subtle top-border gradient line that matches the card's specific "color intent" (e.g., red for flagged, green for safe). It should elevate and glow slightly on hover.

### 3. `Dashboard.jsx` (Page)
- **Purpose**: The main overview landing page.
- **Content**: 
  - A grid of 4 `StatCard`s at the top.
  - A 2-column layout below it: 
    - Left side: "Model Performance" showing horizontal progress bars for Accuracy, Precision, Recall, etc.
    - Right side: A `recharts` Pie chart showing "Risk Distribution" (Safe vs. Low vs. High vs. Critical risk).
  - A bottom full-width section containing a data table of the "Top Flagged Players" (ID, Risk Score, Predicted Label, Total Sent).
- **Requirement**: Wrap the charts in beautiful frosted-glass panels. The data table should have hover effects on rows and use custom badges (pills) for the predicted labels.

### 4. `Players.jsx` (Page)
- **Purpose**: A paginated, searchable database of all players.
- **Content**: 
  - A top control bar with a search input (by Player ID) and filter pills ("All", "Fraudulent", "Safe").
  - A large data table showing all player details (Risk Score with a visual mini-bar, Ground Truth, Total Sent/Received, Ratio).
  - A pagination component at the bottom (Previous/Next buttons, "Page 1 of 100").
- **Requirement**: Make the search bar look like a modern command palette input. The risk score column should render a horizontal bar that fills up corresponding to the percentage, changing color from green to red based on severity.

### 5. `Analytics.jsx` (Page)
- **Purpose**: Deep-dive charts.
- **Content**: 
  - A bar chart comparing all trades vs fraudulent trades across different categories (ACH, Cheque, Wire).
  - An area chart showing transaction volume over time overlaid with fraud cases.
  - A summary grid showing the Classification Report (Precision/Recall side-by-side comparison).
- **Requirement**: Make the `recharts` tooltips look custom and sleek. The grid for the classification report should use glowing typography to emphasize high performance.

### 6. `Sandbox.jsx` (Page)
- **Purpose**: A live, interactive AI visualization demonstrating Graph Neural Network "Message Passing".
- **Content**:
  - Left Side (Controls): 4 Sliders (Total Sent, Total Received, Trades Out, Trades In) to modify the target player's stats. Below that, a "Graph Topology" control panel with buttons to "Add Safe Trade" or "Add Hacker Trade".
  - Right Side (Visualizations): A large central "Live Fraud Risk" gauge or glowing numerical percentage that updates in real-time. Below the risk score, a "Message Passing Visualization" showing a central Target node connected via glowing lines to the neighbor nodes added via the controls.
- **Requirement**: This should be the most visually impressive page. The Risk Score number should have a highly responsive layout (e.g., green when low, pulsing neon red when high). The Topology graph should animate new nodes sliding in, with laser-like connecting lines demonstrating the network flow.

Please generate the full JSX and custom CSS for these components, ensuring they feel incredibly polished, responsive, and ready for a premium enterprise presentation.
