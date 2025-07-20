SELECT 
    income_category,
    COUNT(*) AS num_houses,
    ROUND(AVG(MedHouseVal), 2) AS avg_house_value,
    ROUND(AVG(MedInc), 2) AS avg_income,
    ROUND(AVG(HouseAge), 2) AS avg_house_age,
    ROUND(AVG(AveRooms), 2) AS avg_rooms
FROM 
    housing
GROUP BY 
    income_category
ORDER BY 
    avg_house_value DESC;

SELECT 
    ROUND(Latitude, 2) AS lat,
    ROUND(Longitude, 2) AS lng,
    ROUND(AVG(MedHouseVal), 2) AS avg_house_value,
    COUNT(*) AS num_houses,
    ROUND(AVG(MedInc), 2) AS avg_income,
    ROUND(AVG(HouseAge), 2) AS avg_house_age
FROM 
    housing
GROUP BY 
    ROUND(Latitude, 2), ROUND(Longitude, 2)
HAVING 
    COUNT(*) >= 5
ORDER BY 
    avg_house_value DESC
LIMIT 5;

SELECT 
    income_category,
    ROUND(Latitude, 2) AS lat,
    ROUND(Longitude, 2) AS lng,
    ROUND(AVG(MedHouseVal), 2) AS avg_house_value,
    COUNT(*) AS num_houses,
    SUM(COUNT(*)) OVER (
        PARTITION BY income_category 
        ORDER BY AVG(MedHouseVal) DESC
    ) AS running_total_houses,
    RANK() OVER (
        PARTITION BY income_category 
        ORDER BY AVG(MedHouseVal) DESC
    ) AS rank_in_category
FROM 
    housing
GROUP BY 
    income_category, ROUND(Latitude, 2), ROUND(Longitude, 2)
HAVING 
    COUNT(*) >= 3  -- Only consider areas with at least 3 data points
ORDER BY 
    income_category, rank_in_category;

-- Query 4: Perform a multi-table join to enrich the dataset
-- This query joins the housing data with region metadata to add regional information
SELECT 
    h.income_category,
    r.region_name,
    r.county,
    r.climate_zone,
    CASE WHEN r.coastal = 1 THEN 'Coastal' ELSE 'Inland' END AS location_type,
    COUNT(*) AS num_houses,
    ROUND(AVG(h.MedHouseVal), 2) AS avg_house_value,
    ROUND(AVG(h.MedInc), 2) AS avg_income
FROM 
    housing h
JOIN 
    region_metadata r
ON 
    ROUND(h.Latitude, 2) = ROUND(r.latitude, 2) AND 
    ROUND(h.Longitude, 2) = ROUND(r.longitude, 2)
GROUP BY 
    h.income_category, r.region_name, r.county, r.climate_zone, r.coastal
ORDER BY 
    avg_house_value DESC;

-- Query 5: Advanced analysis using window functions
-- This query calculates percentiles of house values within each climate zone
SELECT 
    r.climate_zone,
    r.region_name,
    ROUND(AVG(h.MedHouseVal), 2) AS avg_house_value,
    ROUND(
        AVG(h.MedHouseVal) / MAX(AVG(h.MedHouseVal)) OVER (PARTITION BY r.climate_zone) * 100, 
        2
    ) AS percent_of_max_in_zone,
    ROUND(
        PERCENT_RANK() OVER (PARTITION BY r.climate_zone ORDER BY AVG(h.MedHouseVal)), 
        2
    ) AS percentile_in_zone,
    COUNT(*) AS num_houses
FROM 
    housing h
JOIN 
    region_metadata r
ON 
    ROUND(h.Latitude, 2) = ROUND(r.latitude, 2) AND 
    ROUND(h.Longitude, 2) = ROUND(r.longitude, 2)
GROUP BY 
    r.climate_zone, r.region_name
ORDER BY 
    r.climate_zone, avg_house_value DESC;

-- Query 6: Impact of indexing demonstration
-- First, let's analyze the query execution plan without using the index
EXPLAIN QUERY PLAN
SELECT 
    income_category,
    COUNT(*) AS num_houses,
    ROUND(AVG(MedHouseVal), 2) AS avg_house_value
FROM 
    housing
WHERE 
    MedHouseVal > 3.0
GROUP BY 
    income_category;

-- Now, let's create an additional index to optimize this query
CREATE INDEX IF NOT EXISTS idx_housing_income_medhouseval 
ON housing(income_category, MedHouseVal);

-- And analyze the query execution plan with the new index
EXPLAIN QUERY PLAN
SELECT 
    income_category,
    COUNT(*) AS num_houses,
    ROUND(AVG(MedHouseVal), 2) AS avg_house_value
FROM 
    housing
WHERE 
    MedHouseVal > 3.0
GROUP BY 
    income_category;

-- The difference in execution plans demonstrates how proper indexing
-- can improve query performance, especially for filtering and grouping operations.
