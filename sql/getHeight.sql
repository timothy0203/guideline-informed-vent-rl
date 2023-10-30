-- This query extracts heights for adult ICU patients.
-- It uses all information from the patient's first ICU day.
-- This is done for consistency with other queries - it's not necessarily needed.
-- Height is unlikely to change throughout a patient's stay.

-- ** Requires the echodata view, generated by concepts/echo-data.sql

DROP MATERIALIZED VIEW IF EXISTS getHeight2 CASCADE;
CREATE MATERIALIZED VIEW getHeight2 as

-- staging table to ensure all heights are in centimeters
with ce0 as
(
    SELECT
      c.icustay_id
      , case
        -- convert inches to centimetres
          when itemid in (920, 1394, 4187, 3486, 226707)
              then valuenum * 2.54
            else valuenum
        end as Height
    FROM mimiciii.chartevents c
    inner join mimiciii.icustays ie
        on c.icustay_id = ie.icustay_id
    WHERE c.valuenum IS NOT NULL
    AND c.itemid in (226730,920, 1394, 4187, 3486,3485,4188) -- height
    AND c.valuenum != 0
    -- exclude rows marked as error
    AND (c.error IS NULL OR c.error = 0)
)
, ce as
(
    SELECT
        icustay_id
        -- extract the median height from the chart to add robustness against outliers
        , AVG(height) as Height_chart
    from ce0
    where height > 100
    group by icustay_id
)
-- requires the echo-data.sql query to run
-- this adds heights from the free-text echo notes
, echo as
(
    select
        ec.subject_id
        -- all echo heights are in inches
        , 2.54*AVG(height) as Height_Echo
    from mimiciii.echo_data ec
    inner join mimiciii.icustays ie
        on ec.subject_id = ie.subject_id
    where height is not null
    and height*2.54 > 100
    group by ec.subject_id
)
select
    ie.icustay_id
    , ie.subject_id
    , coalesce(ce.Height_chart, ec.Height_Echo) as height

    -- components
    , ce.height_chart
    , ec.height_echo
FROM mimiciii.icustays ie

-- filter to only adults
inner join mimiciii.patients pat
    on ie.subject_id = pat.subject_id
    and ie.intime > mimiciii.DATETIME_ADD(pat.dob, INTERVAL '1 YEAR')

left join ce
    on ie.icustay_id = ce.icustay_id

left join echo ec
    on ie.subject_id = ec.subject_id;
