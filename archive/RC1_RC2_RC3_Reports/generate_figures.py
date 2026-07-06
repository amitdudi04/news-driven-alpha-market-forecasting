import os

def write_mermaid(filename, title, content):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# {title}\n\n```mermaid\n{content.strip()}\n```\n")

def generate_figures():
    figures_dir = os.path.join('docs', 'figures')
    
    write_mermaid(os.path.join(figures_dir, 'architecture_diagram.md'), "System Architecture", """
graph TD
    A[GDELT News Stream] -->|module1| B[FinBERT Sentiment]
    C[Yahoo Finance] -->|module3| D[CSI 300 Market Data]
    B -->|module4| E[Feature Engineering]
    D -->|module4| E
    E --> F[Immutable Scaler]
    F -->|module12| G[XGBoost Regime Ensemble]
    G --> H[Meta-Model Confidence Scalar]
    H -->|module13| I{SAFE MODE Check}
    I -->|Z > 15| J[HALT TRADING]
    I -->|Z <= 15| K[Execution Signal]
    """)

    write_mermaid(os.path.join(figures_dir, 'pipeline_flowchart.md'), "Pipeline Flowchart", """
flowchart LR
    A(Daily Trigger) --> B[Ingest]
    B --> C[Clean]
    C --> D[Extract Features]
    D --> E[Scale]
    E --> F[Predict]
    F --> G[Threshold]
    G --> H(Final Signal)
    """)

    write_mermaid(os.path.join(figures_dir, 'module_dependency_graph.md'), "Module Dependency Graph", """
graph TD
    M1[Module 1: News] --> M2[Module 2: Sentiment]
    M3[Module 3: Market] --> M4[Module 4: Features]
    M2 --> M4
    M4 --> M12[Module 12: Inference]
    M12 --> M13[Module 13: Signal Engine]
    M13 --> M14[Module 14: Monitoring]
    """)

    write_mermaid(os.path.join(figures_dir, 'folder_tree.md'), "Folder Tree", """
graph TD
    Root[news-driven-alpha] --> A[config/]
    Root --> B[data/]
    Root --> C[docs/]
    Root --> D[history/]
    Root --> E[models/]
    Root --> F[observability/]
    Root --> G[deployment/]
    Root --> H[app.py]
    Root --> I[run_daily_pipeline.py]
    """)

    write_mermaid(os.path.join(figures_dir, 'execution_lifecycle.md'), "Execution Lifecycle", """
stateDiagram-v2
    [*] --> Ingestion
    Ingestion --> FeatureGen : Valid Data
    Ingestion --> SAFE_MODE : Stale Data > 1 Day
    FeatureGen --> Inference : Z-Score < 15
    FeatureGen --> SAFE_MODE : Z-Score >= 15
    Inference --> Execution : Confidence > 0.3
    Inference --> NoTrade : Confidence < 0.3
    Execution --> [*]
    NoTrade --> [*]
    SAFE_MODE --> [*]
    """)

    write_mermaid(os.path.join(figures_dir, 'training_workflow.md'), "Training Workflow", """
graph TD
    A[Historical Data] --> B[Chronological Split]
    B --> C[Train Partition 80%]
    B --> D[Validation Partition 10%]
    B --> E[Test Partition 10%]
    C --> F[Fit Scaler]
    F --> G[Transform Val/Test]
    F --> H[Train High-Vol & Low-Vol XGBoost]
    H --> I[Train Meta-Model]
    I --> J[model_candidate.pkl]
    """)

    write_mermaid(os.path.join(figures_dir, 'inference_workflow.md'), "Inference Workflow", """
graph TD
    A[New Data] --> B[Load live_scaler.pkl]
    B --> C[Scale Features]
    C --> D{Volatility Regime}
    D -->|> Median| E[High-Vol Model]
    D -->|< Median| F[Low-Vol Model]
    E --> G[Raw Probability]
    F --> G
    G --> H[Meta-Model Assessment]
    H --> I[Scaled Confidence]
    """)

    write_mermaid(os.path.join(figures_dir, 'paper_trading_workflow.md'), "Paper Trading Workflow", """
graph TD
    A[CRON Job] --> B[run_daily_pipeline.py]
    B --> C[Inference]
    C --> D[Signal Generation]
    D --> E[Write to live_tracking.csv]
    E --> F[Dashboard Display]
    """)

    write_mermaid(os.path.join(figures_dir, 'safe_mode_workflow.md'), "SAFE MODE Workflow", """
stateDiagram-v2
    [*] --> CheckStaleness
    CheckStaleness --> CheckSchema : Gap < MAX_STALE
    CheckStaleness --> ESCALATE : Gap > MAX_STALE
    CheckSchema --> CheckDrift : Columns Match
    CheckSchema --> ESCALATE : Columns Missing
    CheckDrift --> PASS : Max Z-Score < 15
    CheckDrift --> ESCALATE : Max Z-Score >= 15
    PASS --> [*]
    ESCALATE --> HALT
    """)

    write_mermaid(os.path.join(figures_dir, 'shadow_model_workflow.md'), "Shadow Model Workflow", """
graph LR
    A[Candidate Model] -->|Walk Forward| B[OOS Performance]
    C[Live Model] -->|Walk Forward| D[OOS Performance]
    B --> E{Compare Sharpe}
    D --> E
    E -->|Candidate Wins| F[Recommend Upgrade]
    E -->|Live Wins| G[Reject Candidate]
    """)

    write_mermaid(os.path.join(figures_dir, 'research_workflow.md'), "Research Workflow", """
graph TD
    A[Hypothesis] --> B[Jupyter Sandbox]
    B --> C[Feature Engineering]
    C --> D[Backtest]
    D -->|Success| E[Draft Module]
    D -->|Failure| A
    E --> F[Peer Audit]
    F --> G[Commit to Production]
    """)

    write_mermaid(os.path.join(figures_dir, 'deployment_workflow.md'), "Deployment Workflow", """
graph LR
    A[Local Code] --> B[Git Commit]
    B --> C[Docker Build]
    C --> D[Unit Tests]
    D --> E[Integration Tests]
    E --> F[Push to Registry]
    F --> G[Production Pull]
    """)

if __name__ == '__main__':
    generate_figures()
