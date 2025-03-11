#%% Contents
# -----------------------------------------------------------------------------------------------
# ### Framework
# 1. Introduction
# 2. Package & Data Imports
# 3. Clean & Profile Data
# 4. Exercise - Part 1
# 5. Exercise - Part 2
# 6. Exercise - Part 3
# -----------------------------------------------------------------------------------------------


#%% INTRODUCTION
"""
Project Title: Fetch -  Data Analyst Take-Home Assessment
Author: Kristen McCaffrey
Date: March 06, 2025
Description:
    This script analyzes the provided Fetch datasets (users, transactions, products)
    to identify data quality issues, answer SQL questions, and summarize findings for stakeholders.

Assumptions:
    - BIRTH_DATE year values prior to 1925 are assumed to be incorrect and will be dropped.
    - Timezones are not needed when calculating age from BIRTH_DATE or account age from CREATED_DATE.
    - Anyone under the age of 13 will be removed as Fetch is not intended for them.
    - GENDER values that are 'prefer not to say', 'not listed' and 'unknown' will be grouped as missing.
    - FINAL_QUANTITY values '276' are assumed to be missing a decimal.
    - The 954 distinct STORE_NAMES in the data are assumed to be distinct and not misspellings or
    capitalization errors that need to be corrected and grouped.
    - The distinct values for CATEGORY_2, CATEGORY_3, CATEGORY_4, MANUFACTURER, and BRAND are assumed
    to be distinct and not misspellings or capitalization errors that need to be corrected and grouped.
    - CATEGORY_1 value, 'Needs Review', is a placeholder and will be dropped as subsequent values in
    the category hierarchy are all missing.
    - Sales data will be assumed to be in USD.
    - User account age will be calculated based on earliest transaction date.
    - BARCODE from 'products' dataset with teh most information will be kept in the case of duplicated
    values, or it will be the first occurance that is kept. Exercise 3 addresses this issue with stakeholders
    as a gap that could be address for better insights.

Libraries Used:
    - os
    - pandas
    - ydata-profiling
    - numpy
    - datetime
    - dateutil
    - sqlite3
    - matplotlib.pyplot
    - seaborn

Notes:
    - This script uses SQLite for in-memory SQL queries.
    - Visualizations are generated to support findings and trends.
"""


#%% PACKAGE & DATA IMPORT

import os
import pandas as pd
from ydata_profiling import ProfileReport
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

# Set current working directory and define folder locations
cwd = os.getcwd()
data_folder = os.path.join(cwd, 'data')
profiling_folder = os.path.join(cwd, 'profiling')
visuals_folder = os.path.join(cwd, 'visuals')

# Import datasets
user = pd.read_csv(os.path.join(data_folder, "USER_TAKEHOME.csv"))
transaction = pd.read_csv(os.path.join(data_folder, "TRANSACTION_TAKEHOME.csv"))
products = pd.read_csv(os.path.join(data_folder, "PRODUCTS_TAKEHOME.csv"))


#%% CLEAN & PROFILE DATA

#### Profiling

# Set parameter to run when profiling reports need to be generated
profiling_run = False

# Profile individual datasets
if profiling_run == True:
    ProfileReport(user).to_file(os.path.join(profiling_folder, "user_profile.html"))
    ProfileReport(transaction).to_file(os.path.join(profiling_folder, "transaction_profile.html"))
    ProfileReport(products).to_file(os.path.join(profiling_folder, "products_profile.html"))

#### Data clean & prep

#### -- User data

# Drop duplicate rows
user = user.drop_duplicates()

# Remove birthdays prior to 1925 (100 years)
user = user.loc[(user['BIRTH_DATE'] >= '1925-01-01')]

# Group extraneous values and convert to missing
texts_to_replace = ['prefer_not_to_say', 'Prefer not to say', 'unknown', 'not_listed', 'not_specified', "My gender isn't listed"]
user['GENDER'] = user['GENDER'].replace(texts_to_replace, np.nan)

# Group different formats together
texts_to_replace = ['non_binary', 'Non-Binary']
user['GENDER'] = user['GENDER'].replace(texts_to_replace, "non_binary")

# Add age column
# Ensure birthdate column is datetime
user['BIRTH_DATE'] = pd.to_datetime(user['BIRTH_DATE'], errors='coerce')

# Birthday function from current date, removing timezone
def calculate_age_precise(birthdate):
    # Drop timezone
    today = datetime.now().replace(tzinfo=None)
    birthdate = birthdate.replace(tzinfo=None)

    return relativedelta(today, birthdate).years

user['AGE'] = user['BIRTH_DATE'].apply(calculate_age_precise)

# Drop users under age 13
user = user.loc[(user['AGE'] >= 13)]

# Add account age column (in months)
# Ensure created date column is datetime
user['CREATED_DATE'] = pd.to_datetime(user['CREATED_DATE'], errors='coerce')

# Account age fucntion from current date (in months), removing timezone
def calculate_account_age_months(creation_date):
    # Drop timezone
    today = datetime.now().replace(tzinfo=None)
    creation_date = creation_date.replace(tzinfo=None)

    rd = relativedelta(today, creation_date)
    return rd.years * 12 + rd.months

# Assuming 'user' is your DataFrame and 'CREATION_DATE' is the column
user['ACCOUNT_AGE_MONTHS'] = user['CREATED_DATE'].apply(calculate_account_age_months)


#### -- Transaction data

# Drop duplicate rows
transaction = transaction.drop_duplicates()

# Convert FINAL_QUANTITY to numeric
transaction['FINAL_QUANTITY'] = transaction['FINAL_QUANTITY'].replace({'zero': '0'})
transaction['FINAL_QUANTITY'] = transaction['FINAL_QUANTITY'].astype(float)

# Fix (assumed) error value in FINAL_QUANTITY
transaction['FINAL_QUANTITY'] = transaction['FINAL_QUANTITY'].replace({276: 2.76})

# Convert FINAL_SALE to numeric
transaction['FINAL_SALE'] = pd.to_numeric(transaction['FINAL_SALE'], errors='coerce')

# Set BARCODE to string
transaction['BARCODE'] = transaction['BARCODE'].astype(str)

# Remove duplicated rows where FINAL_SALE is missing
def remove_duplicate_missing_sales(df):
    # Create a subset excluding FINAL_SALES
    subset_no_sales = df.drop('FINAL_SALE', axis=1)

    # Identify duplicate rows based on the subset
    duplicates = subset_no_sales.duplicated(keep=False)

    # Filter out duplicates with missing sale
    valid_sales_duplicates = df[duplicates & df['FINAL_SALE'].notna()]

    # Filter out non-duplicates
    non_duplicates = df[~duplicates]

    # Combine valid duplicates and non-duplicates
    cleaned_df = pd.concat([non_duplicates, valid_sales_duplicates]).sort_index()

    return cleaned_df

transaction = remove_duplicate_missing_sales(transaction)


#### -- Products data

# Drop duplicate rows
products = products.drop_duplicates()

# Verify CATEGORY_1 values are unique
# print('Uniques values for Category 1: ', products['CATEGORY_1'].unique())

# Explore 'Needs Review' rows
products_needs_review = products.loc[(products['CATEGORY_1'] == 'Needs Review')]
# Drop since other values in the hierarchy (Cat 2, 3, 4) are all null
# print('Uniques values for Category 2:', products_needs_review['CATEGORY_2'].unique())
products = products.loc[(products['CATEGORY_1'] != 'Needs Review')]


# Set BARCODE to string
products['BARCODE'] = products['BARCODE'].astype(str)
# Rename BARCODE for merge
products.rename(columns={'BARCODE':'BARCODE_products'}, inplace=True)

# Show where BARCODE has duplicate values
def get_duplicate_barcode_products(products_df):
    # Identify duplicate BARCODEs
    duplicate_barcodes = products_df['BARCODE_products'].duplicated(keep=False)

    # Create the subset DataFrame
    duplicate_products = products_df[duplicate_barcodes]

    return duplicate_products

duplicate_products_df = get_duplicate_barcode_products(products)

# Get the count of missing and duplicate BARCODE values for Exercise 3
# Count duplicate BARCODEs
duplicate_count = duplicate_products_df['BARCODE_products'].nunique()
# Count missing BARCODEs
missing_count = duplicate_products_df['BARCODE_products'].isin([np.nan, 'N/A', '', 'nan']).sum()
print(f"Number of missing BARCODEs: {missing_count}")
print(f"Number of duplicate BARCODEs: {duplicate_count}")

# Drop missing BARCODE rows (address in Exercise - Part 3)
products = products.dropna(subset=["BARCODE_products"])

# Keeping the most complete rows for duplicate BARCODES, else keep first row for simplicity (addressed in Exercise - Part 3)
def keep_most_complete_duplicates(df, duplicate_column):
    # Identify duplicates
    duplicates = df.duplicated(subset=duplicate_column, keep=False)
    duplicate_rows = df[duplicates]
    non_duplicate_rows = df[~duplicates]

    if duplicate_rows.empty:
        return df

    # Count non-missing values
    non_missing_counts = duplicate_rows.notna().sum(axis=1)

    # Add the count to the df
    duplicate_rows['non_missing_count'] = non_missing_counts

    # Group and select the row with the maximum count
    most_complete_rows = duplicate_rows.sort_values('non_missing_count', ascending=False).drop_duplicates(subset=duplicate_column, keep='first')

    # Drop the extra count column
    most_complete_rows = most_complete_rows.drop('non_missing_count', axis=1)

    # Combine with non-duplicates
    result_df = pd.concat([non_duplicate_rows, most_complete_rows]).sort_index()

    return result_df

products = keep_most_complete_duplicates(products, 'BARCODE_products')


#### Merge data

# # Join datasets (replace with the appropriate join logic)
merged_data = pd.merge(transaction, user, left_on='USER_ID', right_on='ID', how = 'left')
merged_data = merged_data.drop('ID', axis=1)
merged_data = pd.merge(merged_data, products, left_on = 'BARCODE', right_on = 'BARCODE_products', how = 'left')
merged_data = merged_data.drop('BARCODE_products', axis=1)


# Profile the joined dataset
if profiling_run == True:
    merged_data.profile_report().to_file(os.path.join(profiling_folder, "merged_data_profile.html"))


# Variables to drop, keep environment clean before moving on to the next section
del texts_to_replace, products_needs_review


#%% EXERCISE - PART 1
"""
The assumptions that have been made for issues with the data quality as follows:
Assumptions:
    - BIRTH_DATE year values prior to 1925 are assumed to be incorrect and will be dropped.
    - Timezones are not needed when calculating age from BIRTH_DATE or account age from CREATED_DATE.
    - Anyone under the age of 13 will be removed as Fetch is not intended for them.
    - GENDER values that are 'prefer not to say', 'not listed' and 'unknown' will be grouped as missing.
    - FINAL_QUANTITY values '276' are assumed to be missing a decimal.
    - The 954 distinct STORE_NAMES in the data are assumed to be distinct and not misspellings or
    capitalization errors that need to be corrected and grouped.
    - The distinct values for CATEGORY_2, CATEGORY_3, CATEGORY_4, MANUFACTURER, and BRAND are assumed
    to be distinct and not misspellings or capitalization errors that need to be corrected and grouped.
    - CATEGORY_1 value, 'Needs Review', is a placeholder and will be dropped as subsequent values in
    the category hierarchy are all missing.
    - Sales data will be assumed to be in USD.
    - User account age will be calculated based on earliest transaction date.
    - BARCODE from 'products' dataset with teh most information will be kept in the case of duplicated
    values, or it will be the first occurance that is kept. Exercise 3 addresses this issue with stakeholders
    as a gap that could be address for better insights.

Missing and duplicate BARCODE values in the 'products' dataset is a challenge in the data. I would like to speak with the product
owner and other team memebers about this gap and see what can be addressed both short term (and other matching data soruces) and
long term (where can we collect more complete data or address data quality issues).
"""

#%% EXERCISE - PART 2

#### Close-ended questions

#### -- Question 1
#  What are the top 5 brands by receipts scanned among users 21 and over?

# SQL in python function
def get_top_5_brands_by_receipts_users_over_21(merged_df):
    # Create an in-memory SQLite database
    conn = sqlite3.connect(':memory:')

    # Load the merged DataFrame into the database
    merged_df.to_sql('merged_table', conn, index=False, if_exists='replace')

    # SQL query
    query = """
    SELECT BRAND
    	,COUNT(DISTINCT RECEIPT_ID) AS receipt_count
    FROM merged_table
    WHERE AGE >= 21
        AND BRAND IS NOT NULL
    GROUP BY BRAND
    ORDER BY receipt_count DESC LIMIT 5;
    """

    # Execute the query and get the results
    result_df = pd.read_sql_query(query, conn)

    # Close the connection
    conn.close()

    return result_df

top_5_brands = get_top_5_brands_by_receipts_users_over_21(merged_data)
print('')
print('The top 5 brands by receipts scanned among users 21 and over are: ', top_5_brands)

#### -- Question 1 results as comment
# The top 5 brands by receipts scanned among users 21 and over are:
#         BRAND                    receipt_count
# 0  LE PETIT MARSEILLAIS             11
# 1           NERDS CANDY              3
# 2                  DOVE              3
# 3               TRIDENT              2
# 4       SOUR PATCH KIDS              2


#### -- Question 2
# What are the top 5 brands by sales among users that have had their account for at least six months?

# SQL in python function
def get_top_5_brands_by_sales_account_over_6_months(merged_df):
    # Create an in-memory SQLite database
    conn = sqlite3.connect(':memory:')

    # Load the merged DataFrame into the database
    merged_df.to_sql('merged_table', conn, index=False, if_exists='replace')

    # SQL query
    query = """
    SELECT BRAND
        ,SUM(FINAL_SALE) AS total_sales
    FROM merged_table
    WHERE ACCOUNT_AGE_MONTHS >= 6
        AND BRAND IS NOT NULL
        AND FINAL_SALE IS NOT NULL
    GROUP BY BRAND
    ORDER BY total_sales DESC LIMIT 5;
    """

    # Execute the query and get the results
    result_df = pd.read_sql_query(query, conn)

    # Close the connection
    conn.close()

    return result_df

top_5_brands = get_top_5_brands_by_sales_account_over_6_months(merged_data)
print('')
print('The top 5 brands by sales among users whos account is 6+ months old are: ', top_5_brands)

#### -- Question 2 results as comment
# The top 5 brands by sales among users whos account is 6+ months old are:
#         BRAND               total_sales
# 0  LE PETIT MARSEILLAIS        96.71
# 1                   CVS        72.00
# 2               TRIDENT        46.72
# 3                  DOVE        42.88
# 4           COORS LIGHT        34.96


#### Open-ended questions

#### -- Questiion 1
# Which is the leading brand in the Dips & Salsa category?

# Assumptions:
"""
Category Definition:
        - Need to define what constitutes the "Dips & Salsa" category. Assume that CATEGORY_2 or CATEGORY_3 (or a combination)
        contains the most accurate categorization.
        - Assume that the category name is exactly "Dips & Salsa". Slight variations will not be counted here.
Leading Brand Metric:
        - Determine what "leading" means. Assume it means the brand with the highest total sales in the category.
"""

# SQL in python function
def get_leading_dips_salsa_brand(merged_df):
    # Create an in-memory SQLite database
    conn = sqlite3.connect(':memory:')

    # Load the merged DataFrame into the database
    merged_df.to_sql('merged_table', conn, index=False, if_exists='replace')

    # SQL query
    query = """
    SELECT BRAND
    	,SUM(FINAL_SALE) AS total_sales
    FROM merged_table
    WHERE (
    		CATEGORY_2 = 'Dips & Salsa'
            OR CATEGORY_3 = 'Dips & Salsa'
		    )
    	AND BRAND IS NOT NULL
        AND FINAL_SALE IS NOT NULL
    GROUP BY BRAND
    ORDER BY total_sales DESC LIMIT 1;
    """

    # Execute the query and get the results
    result_df = pd.read_sql_query(query, conn)

    # Close the connection
    conn.close()

    return result_df

leading_brand = get_leading_dips_salsa_brand(merged_data)
print('')
print('The leading brand in the Dips & Salsa category is: ', leading_brand)

#### -- Question 1 results as comment
# The leading brand in the Dips & Salsa category is:
#     BRAND       total_sales
# 0  TOSTITOS       260.99


#%% EXERCISE - PART 3

# Create visual to attach to email
output_path = os.path.join(visuals_folder, "barcode_issue_visualization.png")

# Function for chart showing the missing BARCODEs distribution by top 5 brands with teh most missing.
def create_polished_missing_by_top_missing_brands_plot(duplicate_products_df, output_path=output_path):
    """Creates a polished plot showing missing BARCODEs distribution by top 5 brands with most missing."""

    # Set seaborn style (no grid)
    sns.set_style("white")

    # Set font
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']

    # Group by brand, count missing BARCODEs, and get top 5
    missing_by_brand = (
        duplicate_products_df.groupby('BRAND')['BARCODE_products']
        .apply(lambda x: x.isin([np.nan, 'N/A', '', 'nan']).sum())
        .nlargest(5)
    )

    # Create the bar chart
    plt.figure(figsize=(10, 6))
    bars = missing_by_brand.plot(kind='bar', color='#300D38', width=0.7)

    # Add numeric labels to the bars
    for bar in bars.patches:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,  # Place labels directly above the bars
            round(height),
            ha='center',
            va='bottom',
            fontsize=10,
            color='black',
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=2)  # Add label background
        )

    plt.title('Top 5 Brands with Most Missing BARCODEs', fontsize=16, fontweight='bold')
    plt.ylabel('Number of Missing BARCODEs', fontsize=12)
    plt.xlabel('Brand', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout(pad=2)
    plt.savefig(output_path, dpi=300)
    plt.close()

# Assuming 'duplicate_products_df' is your DataFrame
create_polished_missing_by_top_missing_brands_plot(duplicate_products_df)
print('')
print(f"Visualization saved to {output_path}")

# Email to the Product Owner

"""
Subject: Action Needed: BARCODE Data Inconsistencies in Product Catalog

Hi [Product Owner Name],

During my routine data review of the product catalog, I've come across a recurring issue with BARCODE data that needs our attention.

Details:
We're seeing a significant number of entries with either missing or duplicate BARCODEs in the products table.

Specifically:

    We've identified 3968 records where the BARCODE is missing entirely.
    Additionally, 28 records show duplicate BARCODEs, but with varying product details.

Impact:
These inconsistencies are causing some headaches when we try to accurately link our transaction data to the product catalog.
This affects our ability to generate reliable reports on sales performance and product categorization.

For example, when I was working on the sales analysis for the last report, I had to make some assumptions to work around the
 duplicate barcodes, which might have introduced some inaccuracies.

Proposed Next Steps:
To resolve this, I'd like to suggest:

Data Source Audit:

    Let's schedule a quick meeting to review our current data sources for product information.
    I'd also like to reach out to our Nielsen contact to see if they can provide any supplemental data or insights to help us fill in the gaps.

Data Validation Improvements:

    We should look into implementing stronger data validation checks during the product onboarding process to prevent these
    issues from happening in the future.
    Perhaps we can standardize the BARCODE entry process to reduce errors.

Visual:
To give you a better picture of the scale of the problem, I've put together a quick visualization to show which brands have the m
[See attached image: barcode_issue_visualization.png]

Could we schedule some time to discuss this further?

Thanks,

Kristen McCaffrey
Senior Data Analyst
"""
