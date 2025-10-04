SELECT 
    ICD9_CODE, SHORT_TITLE, LONG_TITLE
FROM 
    D_ICD_DIAGNOSES
WHERE 
    LOWER(SHORT_TITLE) LIKE '%type 2 diabetes%'
    OR LOWER(SHORT_TITLE) LIKE '%coronary artery disease%'
    OR LOWER(SHORT_TITLE) LIKE '%chronic kidney disease%'
    OR LOWER(SHORT_TITLE) LIKE '%hypertension%'
    OR LOWER(SHORT_TITLE) LIKE '%acute kidney%';
