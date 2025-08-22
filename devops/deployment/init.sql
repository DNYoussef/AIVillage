-- Development database initialization
-- Creates basic tables for AIVillage development

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Agent states table
CREATE TABLE IF NOT EXISTS agent_states (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id VARCHAR(255) NOT NULL,
    agent_type VARCHAR(100) NOT NULL,
    state JSONB NOT NULL DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(agent_id)
);

-- User profiles table
CREATE TABLE IF NOT EXISTS user_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL UNIQUE,
    profile_data JSONB NOT NULL DEFAULT '{}',
    preferences JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Learning sessions table
CREATE TABLE IF NOT EXISTS learning_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    agent_id VARCHAR(255),
    session_data JSONB NOT NULL DEFAULT '{}',
    metrics JSONB DEFAULT '{}',
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ended_at TIMESTAMP WITH TIME ZONE,
    duration_seconds INTEGER,
    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
);

-- Evolution metrics table
CREATE TABLE IF NOT EXISTS evolution_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id VARCHAR(255) NOT NULL,
    generation INTEGER NOT NULL,
    fitness_score DECIMAL(10, 6),
    metrics JSONB DEFAULT '{}',
    parent_ids TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_agent_states_agent_id ON agent_states(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_states_agent_type ON agent_states(agent_type);
CREATE INDEX IF NOT EXISTS idx_user_profiles_user_id ON user_profiles(user_id);
CREATE INDEX IF NOT EXISTS idx_learning_sessions_user_id ON learning_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_learning_sessions_agent_id ON learning_sessions(agent_id);
CREATE INDEX IF NOT EXISTS idx_evolution_metrics_agent_id ON evolution_metrics(agent_id);
CREATE INDEX IF NOT EXISTS idx_evolution_metrics_generation ON evolution_metrics(generation);

-- Insert some sample data for development
INSERT INTO user_profiles (user_id, profile_data, preferences) VALUES
('dev_user_1', '{"name": "Development User"}', '{"theme": "dark", "notifications": true}')
ON CONFLICT (user_id) DO NOTHING;

INSERT INTO agent_states (agent_id, agent_type, state) VALUES
('king_agent_dev', 'king', '{"status": "active", "mode": "coordination"}'),
('sage_agent_dev', 'sage', '{"status": "active", "mode": "analysis"}'),
('magi_agent_dev', 'magi', '{"status": "active", "mode": "research"}')
ON CONFLICT (agent_id) DO NOTHING;
