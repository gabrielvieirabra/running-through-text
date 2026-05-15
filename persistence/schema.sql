-- MVP schema.
-- Slice 1: users, running_profiles, messages.
-- Slice 2: checkins (with `pains` JSONB), injuries.
-- Other tables (workouts, coach_notes) arrive in later slices.

CREATE TABLE users (
    id TEXT PRIMARY KEY,
    name TEXT,
    age INT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE running_profiles (
    user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    weight_kg NUMERIC,
    height_cm NUMERIC,
    experience_level TEXT,
    pace_5k TEXT,
    pace_10k TEXT,
    longest_run_km NUMERIC,
    weekly_days INT,
    goal TEXT,
    terrain_access TEXT,
    hr_resting INT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_messages_user_created ON messages (user_id, created_at);

-- Check-ins: pointwise state report from the runner.
-- `pains` is an embedded JSONB array of {location, severity 0-10} objects
-- (not its own table — always read alongside the check-in).
CREATE TABLE checkins (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    date DATE NOT NULL DEFAULT CURRENT_DATE,
    sleep_quality INT CHECK (sleep_quality BETWEEN 0 AND 10),
    fatigue INT CHECK (fatigue BETWEEN 0 AND 10),
    motivation INT CHECK (motivation BETWEEN 0 AND 10),
    pains JSONB NOT NULL DEFAULT '[]'::jsonb,
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_checkins_user_date ON checkins (user_id, date);

-- Injuries: lifecycle entity, distinct from "today's pain" in checkins.pains.
CREATE TABLE injuries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    side TEXT,
    year INT,
    status TEXT NOT NULL CHECK (status IN ('active', 'resolved')),
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_injuries_user_status ON injuries (user_id, status);
