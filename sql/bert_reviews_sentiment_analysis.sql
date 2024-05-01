select
    Id as review_id
    ,toDateTime(Time) as dt
    ,Score as rating
    ,CASE
        WHEN Score = 1 THEN 'negative'
        WHEN Score = 5 THEN 'positive'
        ELSE 'neutral'
    END AS sentiment
    ,Text as review
from simulator.flyingfood_reviews
order by review_id asc