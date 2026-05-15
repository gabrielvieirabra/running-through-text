# Pace as primary intensity metric, no RPE

`workouts.target_pace` (min/km, free text) is the primary prescription metric. `workouts.zone` (Z1–Z5) is an optional annotation. **RPE (Rate of Perceived Exertion) does not enter the schema.**

## Why

Concrete pace prescription is more actionable for the runner (with or without a wearable) and more auditable in eval than a subjective self-reported metric. Zones complement pace when the runner has a physiological reference (HR, watch). RPE was considered for being universal and equipment-independent but rejected: it introduces subjective noise into the product's primary metric and duplicates the "effort" signal that already comes from the check-in (`fatigue`, `soreness`).

## Considered options

- **RPE 1-10 primary, pace complementary** — rejected: subjective, noisy, redundant with check-in.
- **Pace + zone both mandatory** — rejected: not every runner has calibrated zones.
- **Pace primary + zone optional** *(chosen)*.

## Consequences

- For runners without a watch/zones: agent prescribes pace only, no zone.
- For `fartlek`, `ladeira`, `intervalado`: `target_pace` accepts a pace plan as text (e.g. `4:00/km work / 6:00/km rest`).
- If real adoption pain shows up among feel-based runners ("I have no watch, I don't know my pace"), this decision is revisited — RPE could become an additional field, never replacing pace.
