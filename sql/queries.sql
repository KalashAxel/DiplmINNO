
-- queries.sql
-- Набор запросов для контроля качества, EDA и отчета

-- 1) Объем данных
SELECT count(*) AS n_total FROM raw.credit;

-- 2) Проверка некорректного возраста
SELECT
  count(*) FILTER (WHERE age IS NULL) AS age_null,
  count(*) FILTER (WHERE age < 18) AS age_lt_18,
  count(*) FILTER (WHERE age > 120) AS age_gt_120
FROM raw.credit;

-- 3) Баланс классов (после витрины признаков)
SELECT SeriousDlqin2yrs,
       count(*) AS n,
       round(100.0*count(*)/sum(count(*)) OVER (), 2) AS pct
FROM features.credit
GROUP BY SeriousDlqin2yrs
ORDER BY SeriousDlqin2yrs;

-- 4) Размеры выборок
SELECT split, count(*) AS n
FROM features.credit
GROUP BY split
ORDER BY split;

-- 5) Баланс классов по выборкам (контроль стратификации)
SELECT split, SeriousDlqin2yrs, count(*) AS n
FROM features.credit
GROUP BY split, SeriousDlqin2yrs
ORDER BY split, SeriousDlqin2yrs;

-- 6) Доли пропусков (через индикаторы)
SELECT
  avg(is_missing_MonthlyIncome)::numeric(6,4) AS miss_income_rate,
  avg(is_missing_NumberOfDependents)::numeric(6,4) AS miss_dep_rate
FROM features.credit;

-- 7) Быстрая sanity-проверка клипа (минимумы/максимумы уже после обрезки)
SELECT
  min(RevolvingUtilizationOfUnsecuredLines) AS ru_min,
  max(RevolvingUtilizationOfUnsecuredLines) AS ru_max,
  min(DebtRatio) AS dr_min,
  max(DebtRatio) AS dr_max,
  min(MonthlyIncome) AS mi_min,
  max(MonthlyIncome) AS mi_max
FROM features.credit;-- 8) Распределение по возрастным группам
SELECT age_group, count(*) AS n
FROM features.credit
GROUP BY age_group
ORDER BY n DESC;

-- 9) Баланс классов по возрастным группам (важно для анализа недискриминационности)
SELECT age_group, SeriousDlqin2yrs, count(*) AS n
FROM features.credit
GROUP BY age_group, SeriousDlqin2yrs
ORDER BY age_group, SeriousDlqin2yrs;
