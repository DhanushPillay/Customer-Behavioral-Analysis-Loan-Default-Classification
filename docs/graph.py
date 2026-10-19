graph TD
    A[Raw Source Data<br><i>loan, trans, client, account, disp</i>] --> B[Data Integration & Joins<br>Merge relational tables via keys]
    B --> C[Temporal Filtering<br>Remove all transactions on/after loan date]
    C --> D[Missing Value Treatment<br>Impute & handle nulls]
    D --> E[Feature Engineering<br>Extract 13 pre-loan behavioral metrics]
    E --> F[Train-Test Split & Scaling<br>StandardScaler strictly inside CV folds]
    F --> G[Clean Modeling-Ready Data<br>Ready for Classification & Clustering]
    
    style A fill:#f3f4f6,stroke:#374151,stroke-width:2px
    style G fill:#d1fae5,stroke:#059669,stroke-width:2px