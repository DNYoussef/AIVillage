graph TD
    subgraph "CODEX System"
        subgraph "Core Services"
            A[LibP2P Mesh Network] --> B(Agents)
            C[Evolution Manager] --> D(King Agent)
            E[Wikipedia STORM Pipeline] --> F(Unified RAG Management)
            G[Digital Twin API] --> H{React Native App}
            I[VILLAGE Token] --> J(Education Platform)
        end

        subgraph "Agents"
            B --> K(King Agent)
            B --> L(Magi Agent)
            B --> M(Sage Agent)
            F --> M
        end

        subgraph "Data"
            N[Wikipedia Dataset] --> E
        end
    end