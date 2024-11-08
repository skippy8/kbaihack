# AWS tool - EMR Studio with workspace with jupyter notebook was used for pre-preparation of analysis datasets

# initializing database catalog 
spark.sql("USE `396608780683-kb-hackathon-glue-database-catalog`")
spark.sql("SHOW TABLES").show(truncate=False)

import pandas as pd

df_campa = spark.sql("SELECT identity_id_hash, contact_status_exported_first_date, campaign_planning_name, offer_category_code, offer_name, product_l2, channel_name, nbi_fictive FROM `396608780683-kb-hackathon-glue-database-catalog`.campa where product_l2 <> 'Produkt_L2_01' and product_l2 <> 'Produkt_L2_02'")

df_ib = spark.sql("SELECT identity_id_hash, event_id, event_timestamp, event_type, screenview_flag, session_id, page_url_pseudonymized, referrer_url_pseudonymized, session_time, content_title, event_name FROM `396608780683-kb-hackathon-glue-database-catalog`.ib")

df_prodeje = spark.sql("SELECT identity_id_hash, agreement_id_hash, date_valid, product_l1, product_l2, product_l3,business_channel_name, ndb_flag,CASE WHEN nbi_fictive < 2100 THEN '<2100'WHEN nbi_fictive > 2100 THEN '>2100'END AS nbi_fictive_flag FROM `396608780683-kb-hackathon-glue-database-catalog`.prodeje where product_l2 <> 'Produkt_L2_01' and product_l2 <> 'Produkt_L2_02'")

df_web = spark.sql("SELECT identity_id_hash, event_id, event_timestamp, event_type, screenview_flag, session_id, user_id, referrer_url_pseudonymized, page_name_pseudonymized, utm_campaign_pseudonymized, os_family, os_version, device_type, browser_family, is_touch_capable, city_name, app_language, mktg2_product_code_pseudonymized, link_text_pseudonymized, link_url_pseudonymized, menu_category_pseudonymized,entry_page_url_pseudonymized, entry_page_name_pseudonymized, exit_page_url_pseudonymized, exit_page_name_pseudonymized, nonzero_screen_flag, nonzero_session_flag, engaged_session_flag, bounce_session_flag, channel_name, utm_term, utm_medium, utm_content, utm_source, lead, content_title, component_type, internal_prom_target_url_pseudonymized FROM `396608780683-kb-hackathon-glue-database-catalog`.web9 UNION SELECT identity_id_hash, event_id, event_timestamp, event_type, screenview_flag, session_id, user_id, referrer_url_pseudonymized, page_name_pseudonymized, utm_campaign_pseudonymized, os_family, os_version, device_type, browser_family, is_touch_capable, city_name, app_language, mktg2_product_code_pseudonymized, link_text_pseudonymized, link_url_pseudonymized, menu_category_pseudonymized,entry_page_url_pseudonymized, entry_page_name_pseudonymized, exit_page_url_pseudonymized, exit_page_name_pseudonymized, nonzero_screen_flag, nonzero_session_flag, engaged_session_flag, bounce_session_flag, channel_name, utm_term, utm_medium, utm_content, utm_source, lead, content_title, component_type, internal_prom_target_url_pseudonymized FROM `396608780683-kb-hackathon-glue-database-catalog`.web8 UNION SELECT identity_id_hash, event_id, event_timestamp, event_type, screenview_flag, session_id, user_id, referrer_url_pseudonymized, page_name_pseudonymized, utm_campaign_pseudonymized, os_family, os_version, device_type, browser_family, is_touch_capable, city_name, app_language, mktg2_product_code_pseudonymized, link_text_pseudonymized, link_url_pseudonymized, menu_category_pseudonymized,entry_page_url_pseudonymized, entry_page_name_pseudonymized, exit_page_url_pseudonymized, exit_page_name_pseudonymized, nonzero_screen_flag, nonzero_session_flag, engaged_session_flag, bounce_session_flag, channel_name, utm_term, utm_medium, utm_content, utm_source, lead, content_title, component_type, internal_prom_target_url_pseudonymized FROM `396608780683-kb-hackathon-glue-database-catalog`.web7")



df_app = spark.sql("SELECT identity_id_hash, event_id, event_timestamp, event_type,screenview_flag, session_id, page_name_pseudonymized, session_time, calc_page_name, calc_page_name, calc_prev_page_name, calc_next_page_name, previous_session_id, content_title, device_manufacturer FROM `396608780683-kb-hackathon-glue-database-catalog`.app7 UNION SELECT identity_id_hash, event_id, event_timestamp, event_type,screenview_flag, session_id, page_name_pseudonymized, session_time, calc_page_name, calc_page_name, calc_prev_page_name, calc_next_page_name, previous_session_id, content_title, device_manufacturer FROM `396608780683-kb-hackathon-glue-database-catalog`.app8 UNION SELECT identity_id_hash, event_id, event_timestamp, event_type,screenview_flag, session_id, page_name_pseudonymized, session_time, calc_page_name, calc_page_name, calc_prev_page_name, calc_next_page_name, previous_session_id, content_title, device_manufacturer FROM `396608780683-kb-hackathon-glue-database-catalog`.app9_1 UNION SELECT identity_id_hash, event_id, event_timestamp, event_type,screenview_flag, session_id, page_name_pseudonymized, session_time, calc_page_name, calc_page_name, calc_prev_page_name, calc_next_page_name, previous_session_id, content_title, device_manufacturer FROM `396608780683-kb-hackathon-glue-database-catalog`.app9_2")


# Then we joined these tables using Querry editor 


WITH campa AS (
  SELECT identity_id_hash,
    contact_status_exported_first_date,
    campaign_planning_name,
    offer_category_code,
    offer_name,
    product_l2,
    channel_name,
    nbi_fictive
  FROM "campa"
  where product_l2 <> 'Produkt_L2_01'
    and product_l2 <> 'Produkt_L2_02'
),
prodeje AS (
  SELECT identity_id_hash,
    agreement_id_hash,
    date_valid,
    product_l1,
    product_l2,
    product_l3,
    business_channel_name,
    ndb_flag,
    CASE
      WHEN nbi_fictive < 2100 THEN '<2100'
      WHEN nbi_fictive > 2100 THEN '>2100'
    END AS nbi_fictive_flag
  FROM "prodeje"
  where product_l2 <> 'Produkt_L2_01'
    and product_l2 <> 'Produkt_L2_02'
),
ib AS (
  SELECT identity_id_hash,
    event_id,
    event_timestamp,
    event_type,
    screenview_flag,
    session_id,
    page_url_pseudonymized,
    referrer_url_pseudonymized,
    session_time,
    content_title,
    event_name
  FROM "ib"
),
web AS (
  SELECT identity_id_hash,
    event_id,
    event_timestamp,
    event_type,
    screenview_flag,
    session_id,
    user_id,
    referrer_url_pseudonymized,
    page_name_pseudonymized,
    utm_campaign_pseudonymized,
    os_family,
    os_version,
    device_type,
    browser_family,
    is_touch_capable,
    city_name,
    app_language,
    mktg2_product_code_pseudonymized,
    link_text_pseudonymized,
    link_url_pseudonymized,
    menu_category_pseudonymized,
    entry_page_url_pseudonymized,
    entry_page_name_pseudonymized,
    exit_page_url_pseudonymized,
    exit_page_name_pseudonymized,
    nonzero_screen_flag,
    nonzero_session_flag,
    engaged_session_flag,
    bounce_session_flag,
    channel_name,
    utm_term,
    utm_medium,
    utm_content,
    utm_source,
    lead,
    content_title,
    component_type,
    internal_prom_target_url_pseudonymized
  FROM "web9"
  UNION
  SELECT identity_id_hash,
    event_id,
    event_timestamp,
    event_type,
    screenview_flag,
    session_id,
    user_id,
    referrer_url_pseudonymized,
    page_name_pseudonymized,
    utm_campaign_pseudonymized,
    os_family,
    os_version,
    device_type,
    browser_family,
    is_touch_capable,
    city_name,
    app_language,
    mktg2_product_code_pseudonymized,
    link_text_pseudonymized,
    link_url_pseudonymized,
    menu_category_pseudonymized,
    entry_page_url_pseudonymized,
    entry_page_name_pseudonymized,
    exit_page_url_pseudonymized,
    exit_page_name_pseudonymized,
    nonzero_screen_flag,
    nonzero_session_flag,
    engaged_session_flag,
    bounce_session_flag,
    channel_name,
    utm_term,
    utm_medium,
    utm_content,
    utm_source,
    lead,
    content_title,
    component_type,
    internal_prom_target_url_pseudonymized
  FROM "web8"
  UNION
  SELECT identity_id_hash,
    event_id,
    event_timestamp,
    event_type,
    screenview_flag,
    session_id,
    user_id,
    referrer_url_pseudonymized,
    page_name_pseudonymized,
    utm_campaign_pseudonymized,
    os_family,
    os_version,
    device_type,
    browser_family,
    is_touch_capable,
    city_name,
    app_language,
    mktg2_product_code_pseudonymized,
    link_text_pseudonymized,
    link_url_pseudonymized,
    menu_category_pseudonymized,
    entry_page_url_pseudonymized,
    entry_page_name_pseudonymized,
    exit_page_url_pseudonymized,
    exit_page_name_pseudonymized,
    nonzero_screen_flag,
    nonzero_session_flag,
    engaged_session_flag,
    bounce_session_flag,
    channel_name,
    utm_term,
    utm_medium,
    utm_content,
    utm_source,
    lead,
    content_title,
    component_type,
    internal_prom_target_url_pseudonymized
  FROM "web7"
),
app AS (
  SELECT identity_id_hash,
    event_id,
    event_timestamp,
    event_type,
    screenview_flag,
    session_id,
    page_name_pseudonymized,
    session_time,
    calc_page_name,
    calc_page_name,
    calc_prev_page_name,
    calc_next_page_name,
    previous_session_id,
    content_title,
    device_manufacturer
  FROM "app7"
  UNION
  SELECT identity_id_hash,
    event_id,
    event_timestamp,
    event_type,
    screenview_flag,
    session_id,
    page_name_pseudonymized,
    session_time,
    calc_page_name,
    calc_page_name,
    calc_prev_page_name,
    calc_next_page_name,
    previous_session_id,
    content_title,
    device_manufacturer
  FROM "app8"
  UNION
  SELECT identity_id_hash,
    event_id,
    event_timestamp,
    event_type,
    screenview_flag,
    session_id,
    page_name_pseudonymized,
    session_time,
    calc_page_name,
    calc_page_name,
    calc_prev_page_name,
    calc_next_page_name,
    previous_session_id,
    content_title,
    device_manufacturer
  FROM "app9_1"
  UNION
  SELECT identity_id_hash,
    event_id,
    event_timestamp,
    event_type,
    screenview_flag,
    session_id,
    page_name_pseudonymized,
    session_time,
    calc_page_name,
    calc_page_name,
    calc_prev_page_name,
    calc_next_page_name,
    previous_session_id,
    content_title,
    device_manufacturer
  FROM "app9_2"
)
SELECT *
FROM web AS a
  LEFT JOIN app AS b ON a.identity_id_hash = b.identity_id_hash
  LEFT JOIN prodeje AS c ON a.identity_id_hash = c.identity_id_hash
  LEFT JOIN ib AS d ON a.identity_id_hash = d.identity_id_hash
  LEFT JOIN campa AS e ON a.identity_id_hash = e.identity_id_hash
 
  ;


# Further implementation of datasets and modeling were made using google colab
