SELECT
    main.user_id as user_id,
    main.item_id as item_id,
    SUM(main.units) AS qty,
    avg_price.price as price
FROM
    default.karpov_express_orders AS main
JOIN
    (SELECT
         item_id AS avg_item_id,
         ROUND(AVG(price), 2) AS price
     FROM default.karpov_express_orders
     WHERE toDate(timestamp) BETWEEN %(start_date)s AND %(end_date)s
     GROUP BY item_id
    ) AS avg_price
ON main.item_id = avg_price.avg_item_id
WHERE
    toDate(main.timestamp) BETWEEN %(start_date)s AND %(end_date)s
GROUP BY
    main.user_id, main.item_id, avg_price.price
ORDER BY
    main.user_id, main.item_id