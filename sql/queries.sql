-- queries.sql

-- Баланс классов
SELECT SeriousDlqin2yrs,
       count(*) AS n,
       round(100.0*count(*)/sum(count(*)) OVER (), 2) AS pct
FROM features.credit
GROUP BY SeriousDlqin2yrs
ORDER BY SeriousDlqin2yrs;

-- Размеры выборок
SELECT split, count(*) AS n
FROM features.credit
GROUP BY split
ORDER BY split;

-- Баланс классов по выборкам
SELECT split, SeriousDlqin2yrs, count(*) AS n
FROM features.credit
GROUP BY split, SeriousDlqin2yrs
ORDER BY split, SeriousDlqin2yrs;

-- Доли пропусков (через индикаторы)
SELECT
  avg(is_missing_MonthlyIncome)::numeric(6,4) AS miss_income_rate,
  avg(is_missing_NumberOfDependents)::numeric(6,4) AS miss_dep_rate
FROM features.credit;

-- Распределение по возрастным группам
SELECT age_group, count(*) AS n
FROM features.credit
GROUP BY age_group
ORDER BY n DESC;
