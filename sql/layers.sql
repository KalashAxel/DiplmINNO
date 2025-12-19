-- layers.sql
-- Слои: raw -> cleaned -> features

DROP TABLE IF EXISTS raw.credit;

CREATE TABLE raw.credit (
    id BIGINT,
    SeriousDlqin2yrs INT,
    RevolvingUtilizationOfUnsecuredLines DOUBLE PRECISION,
    age INT,
    NumberOfTime30_59DaysPastDueNotWorse INT,
    DebtRatio DOUBLE PRECISION,
    MonthlyIncome DOUBLE PRECISION,
    NumberOfOpenCreditLinesAndLoans INT,
    NumberOfTimes90DaysLate INT,
    NumberRealEstateLoansOrLines INT,
    NumberOfTime60_89DaysPastDueNotWorse INT,
    NumberOfDependents DOUBLE PRECISION
);

-- Загрузка (пример):
-- COPY raw.credit FROM '/path/to/GiveMeSomeCredit-training.csv' WITH (FORMAT csv, HEADER true);

DROP TABLE IF EXISTS cleaned.credit;

WITH med AS (
    SELECT
        percentile_cont(0.5) WITHIN GROUP (ORDER BY MonthlyIncome) AS med_income,
        percentile_cont(0.5) WITHIN GROUP (ORDER BY NumberOfDependents) AS med_dep
    FROM raw.credit
),
base AS (
    SELECT
        COALESCE(id, row_number() OVER ()) AS id,
        SeriousDlqin2yrs,
        RevolvingUtilizationOfUnsecuredLines,
        age,
        NumberOfTime30_59DaysPastDueNotWorse,
        DebtRatio,
        MonthlyIncome,
        NumberOfOpenCreditLinesAndLoans,
        NumberOfTimes90DaysLate,
        NumberRealEstateLoansOrLines,
        NumberOfTime60_89DaysPastDueNotWorse,
        NumberOfDependents,

        CASE WHEN MonthlyIncome IS NULL THEN 1 ELSE 0 END AS is_missing_MonthlyIncome,
        CASE WHEN NumberOfDependents IS NULL THEN 1 ELSE 0 END AS is_missing_NumberOfDependents,

        COALESCE(MonthlyIncome, (SELECT med_income FROM med)) AS MonthlyIncome_imp,
        COALESCE(NumberOfDependents, (SELECT med_dep FROM med)) AS NumberOfDependents_imp
    FROM raw.credit
    WHERE age IS NOT NULL AND age BETWEEN 18 AND 120
      AND SeriousDlqin2yrs IN (0,1)
),
ranked AS (
    SELECT
        *,
        row_number() OVER (PARTITION BY SeriousDlqin2yrs ORDER BY random()) AS rn,
        count(*)    OVER (PARTITION BY SeriousDlqin2yrs) AS cnt
    FROM base
)
SELECT
    id,
    SeriousDlqin2yrs,

    RevolvingUtilizationOfUnsecuredLines,
    age,
    NumberOfTime30_59DaysPastDueNotWorse,
    DebtRatio,
    MonthlyIncome_imp AS MonthlyIncome,
    NumberOfOpenCreditLinesAndLoans,
    NumberOfTimes90DaysLate,
    NumberRealEstateLoansOrLines,
    NumberOfTime60_89DaysPastDueNotWorse,
    NumberOfDependents_imp AS NumberOfDependents,

    is_missing_MonthlyIncome,
    is_missing_NumberOfDependents,

    CASE
        WHEN rn <= cnt * 0.70 THEN 'train'
        WHEN rn <= cnt * 0.85 THEN 'valid'
        ELSE 'test'
    END AS split
INTO cleaned.credit
FROM ranked;

CREATE INDEX IF NOT EXISTS idx_cleaned_split  ON cleaned.credit(split);
CREATE INDEX IF NOT EXISTS idx_cleaned_target ON cleaned.credit(SeriousDlqin2yrs);

DROP TABLE IF EXISTS features.clip_bounds;
DROP TABLE IF EXISTS features.credit;

CREATE TABLE features.clip_bounds AS
SELECT
    percentile_cont(0.005) WITHIN GROUP (ORDER BY RevolvingUtilizationOfUnsecuredLines) AS ru_lo,
    percentile_cont(0.995) WITHIN GROUP (ORDER BY RevolvingUtilizationOfUnsecuredLines) AS ru_hi,

    percentile_cont(0.005) WITHIN GROUP (ORDER BY DebtRatio) AS dr_lo,
    percentile_cont(0.995) WITHIN GROUP (ORDER BY DebtRatio) AS dr_hi,

    percentile_cont(0.005) WITHIN GROUP (ORDER BY MonthlyIncome) AS mi_lo,
    percentile_cont(0.995) WITHIN GROUP (ORDER BY MonthlyIncome) AS mi_hi
FROM cleaned.credit
WHERE split = 'train';

WITH b AS (SELECT * FROM features.clip_bounds),
x AS (
    SELECT
        c.*,
        LEAST(GREATEST(c.RevolvingUtilizationOfUnsecuredLines, b.ru_lo), b.ru_hi) AS ru_clip,
        LEAST(GREATEST(c.DebtRatio, b.dr_lo), b.dr_hi) AS dr_clip,
        LEAST(GREATEST(c.MonthlyIncome, b.mi_lo), b.mi_hi) AS mi_clip
    FROM cleaned.credit c
    CROSS JOIN b
)
SELECT
    id,
    split,
    SeriousDlqin2yrs,

    CASE
        WHEN age BETWEEN 18 AND 24 THEN '18-24'
        WHEN age BETWEEN 25 AND 34 THEN '25-34'
        WHEN age BETWEEN 35 AND 44 THEN '35-44'
        WHEN age BETWEEN 45 AND 54 THEN '45-54'
        WHEN age BETWEEN 55 AND 64 THEN '55-64'
        ELSE '65+'
    END AS age_group,

    age,
    ru_clip AS RevolvingUtilizationOfUnsecuredLines,
    NumberOfTime30_59DaysPastDueNotWorse,
    dr_clip AS DebtRatio,
    mi_clip AS MonthlyIncome,
    NumberOfOpenCreditLinesAndLoans,
    NumberOfTimes90DaysLate,
    NumberRealEstateLoansOrLines,
    NumberOfTime60_89DaysPastDueNotWorse,
    NumberOfDependents,

    is_missing_MonthlyIncome,
    is_missing_NumberOfDependents,

    LN(1 + GREATEST(mi_clip, 0)) AS log_MonthlyIncome,
    LN(1 + GREATEST(dr_clip, 0)) AS log_DebtRatio,
    LN(1 + GREATEST(ru_clip, 0)) AS log_RevolvingUtilization
INTO features.credit
FROM x;

CREATE INDEX IF NOT EXISTS idx_feat_split    ON features.credit(split);
CREATE INDEX IF NOT EXISTS idx_feat_target   ON features.credit(SeriousDlqin2yrs);
CREATE INDEX IF NOT EXISTS idx_feat_agegroup ON features.credit(age_group);
