-- MVP schema.
-- Slice 1: users, running_profiles, messages.
-- Slice 2: checkins (with `pains` JSONB), injuries.
-- Slice 3: workouts (log of realized runs).
-- Slice 4: running_profiles tightened with experience_level CHECK and a
--          positive `injury_history_acknowledged` flag for onboarding gating.
-- Slice 6: coach_notes (narrative summary, M3 memory per ADR 0001).

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
    experience_level TEXT CHECK (
        experience_level IS NULL
        OR experience_level IN ('iniciante', 'intermediario', 'avancado')
    ),
    pace_5k TEXT,
    pace_10k TEXT,
    longest_run_km NUMERIC,
    weekly_days INT CHECK (weekly_days IS NULL OR weekly_days BETWEEN 1 AND 7),
    goal TEXT,
    terrain_access TEXT,
    hr_resting INT,
    -- Positive signal that the runner addressed injury history during onboarding.
    -- Flipped to TRUE either by `register_injury` (mentioning a past injury counts)
    -- or by `update_profile` when the runner explicitly says "no injuries".
    injury_history_acknowledged BOOLEAN NOT NULL DEFAULT FALSE,
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

-- Workouts: log of REALIZED runs (B1, ADR 0002). Not planned — no `planned_*`
-- columns and no `status`. `type` uses canonical PT-BR enum values (CONTEXT.md
-- "Workout taxonomy"). `target_pace` is free text (ADR 0004): pace is primary,
-- zone is an optional annotation, RPE is deliberately absent.
-- `perceived_effort` here is a post-hoc "how hard was it" log field on the
-- realized workout — NOT RPE-as-intensity-prescription.
CREATE TABLE workouts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    date DATE NOT NULL DEFAULT CURRENT_DATE,
    type TEXT NOT NULL CHECK (type IN (
        'rodagem',
        'longo',
        'regenerativo',
        'fartlek',
        'intervalado',
        'tempo',
        'ladeira',
        'prova',
        'simulado',
        'outro'
    )),
    target_pace TEXT,
    zone TEXT CHECK (zone IS NULL OR zone IN ('Z1', 'Z2', 'Z3', 'Z4', 'Z5')),
    distance_km NUMERIC,
    duration_min NUMERIC,
    perceived_effort INT CHECK (perceived_effort IS NULL OR perceived_effort BETWEEN 0 AND 10),
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_workouts_user_date ON workouts (user_id, date);

-- Coach notes: narrative summary written by the agent itself (ADR 0001 M3).
-- The latest note is loaded as part of every turn's context preamble; older
-- rows are kept for audit but never read by the agent. Triggers map to the
-- mechanics in CONTEXT.md ("coach_note mechanics"): a new workout, a risk
-- flag firing on a check-in, N days of idleness, an explicit `manual` push
-- from the agent via `update_coach_note`, and a `bootstrap` regeneration.
CREATE TABLE coach_notes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    trigger TEXT NOT NULL CHECK (
        trigger IN ('new_workout', 'risk_flag', 'idle', 'manual', 'bootstrap')
    )
);

CREATE INDEX idx_coach_notes_user_generated ON coach_notes (user_id, generated_at DESC);
