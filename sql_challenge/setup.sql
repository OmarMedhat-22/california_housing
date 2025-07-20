CREATE TABLE IF NOT EXISTS housing (
    id INTEGER PRIMARY KEY,
    MedInc REAL,
    HouseAge REAL,
    AveRooms REAL,
    AveBedrms REAL,
    Population REAL,
    AveOccup REAL,
    Latitude REAL,
    Longitude REAL,
    MedHouseVal REAL,
    income_category TEXT,
    age_category TEXT,
    population_density TEXT
);

CREATE INDEX IF NOT EXISTS idx_housing_medhouseval ON housing(MedHouseVal);
CREATE INDEX IF NOT EXISTS idx_housing_income_category ON housing(income_category);

CREATE TABLE IF NOT EXISTS region_metadata (
    region_id INTEGER PRIMARY KEY,
    latitude REAL,
    longitude REAL,
    region_name TEXT,
    county TEXT,
    climate_zone TEXT,
    coastal INTEGER
);
