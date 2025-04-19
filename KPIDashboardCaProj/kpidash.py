"""
KPI Sales Dashboard

This Streamlit application provides an interactive dashboard for visualizing and analyzing sales data.
It connects to a PostgreSQL database, performs various calculations, and displays KPIs, charts,
and forecasts.

Author: Isaac DeVera

"""

import streamlit as st
import pandas as pd
import psycopg2
import plotly.express as px
from datetime import datetime, timedelta
import statsmodels.api as sm
import hashlib  # For password hashing
import os  # For environment variables
import configparser  # For config.ini
from typing import List, Tuple

# --- Configuration ---

# Load configuration from config.ini
config = configparser.ConfigParser()
config.read('config.ini')

# Database connection parameters from config
DB_HOST = os.environ.get('DB_HOST', config['database']['host'])
DB_PORT = os.environ.get('DB_PORT', config['database']['port'])
DB_NAME = os.environ.get('DB_NAME', config['database']['dbname'])
DB_USER = os.environ.get('DB_USER', config['database']['user'])
DB_PASSWORD = os.environ.get('DB_PASSWORD', config['database']['password'])

# Table and column names from config
TABLE_NAME = config['table']['name']
ORDER_DATE_COL = config['columns']['order_date']
PRODUCT_LINE_COL = config['columns']['product_line']
SALES_COL = config['columns']['sales']
ORDER_NUMBER_COL = config['columns']['order_number']
CUSTOMER_NAME_COL = config['columns']['customer_name']
TERRITORY_COL = config['columns']['territory']
COUNTRY_COL = config['columns']['country']
STATE_COL = config['columns']['state']
CITY_COL = config['columns']['city']
STATUS_COL = config['columns']['status']
DEAL_SIZE_COL = config['columns']['deal_size']
PRODUCT_CODE_COL = config['columns']['product_code']
QUANTITY_ORDERED_COL = config['columns']['quantity_ordered']
PRICE_EACH_COL = config['columns']['price_each']
MSRP_COL = config['columns']['msrp']
ORDER_LINE_NUMBER_COL = config['columns']['order_line_number']  # [cite: 1, 2]

# --- User Authentication ---

# User authentication and roles
USERS = {
    "admin": {"password": hashlib.sha256("s4l3s".encode()).hexdigest(), "role": "admin"}
}  # [cite: 2]

def check_password(username, password):
    """
    Checks if the provided password matches the stored hash for the given user.

    Args:
        username (str): The username to check.
        password (str): The password to verify.

    Returns:
        bool: True if the password is correct, False otherwise.
    """
    user = USERS.get(username)
    if user:  # If user exists
        hashed_password = hashlib.sha256(password.encode()).hexdigest()  # Hexadecimal string representing the SHA-256 hash of the password
        return user["password"] == hashed_password  # Check if password matches
    return False  # If user doesn't exist, return False

def authenticate():
    """
    Handles user authentication via a login form.  Uses Streamlit's session state
    to persist authentication status.
    """
    if 'authenticated' not in st.session_state:  # If not authenticated
        st.session_state.authenticated = False  # Not authenticated
        st.session_state.user_role = None  # No role

    if not st.session_state.authenticated:  # If not authenticated, create the login website
        username = st.text_input("Username")  # Create Username space
        password = st.text_input("Password", type="password")  # Create Password space
        if st.button("Login"):  # If Login is pressed
            if check_password(username, password):  # Check password if it matches; if True... [cite: 2, 3, 4]
                st.session_state.authenticated = True  # Authenticated
                st.session_state.user_role = USERS[username]["role"]  # Role established
                st.success("Logged in!")  # Logged in!
                st.button("Go to Dashboard!")  # Creates a button saying "Go to Dashboard!" [cite: 5]
            else:
                st.error("Incorrect username or password")  # Incorrect username or password
        st.stop()  # Stop execution if not authenticated

# --- Database Connection ---

def get_db_connection():
    """
    Connects to the PostgreSQL database using parameters from the configuration.

    Returns:
        psycopg2.connection: A connection object, or None if the connection fails.
    """
    try:
        conn = psycopg2.connect(  # Check to see if connection parameters are correct
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
        )
        return conn
    except psycopg2.Error as e:  # If not correct, say error
        st.error(f"Error connecting to the database: {e}")
        return None  # If nothing, say None [cite: 6, 7]

# --- Data Loading ---

@st.cache_data  # Fetch data (with input sanitization)
def load_data(table_name: str) -> pd.DataFrame:
    """
    Loads data from the specified database table into a Pandas DataFrame.

    Args:
        table_name (str): The name of the table to load data from.

    Returns:
        pd.DataFrame: A DataFrame containing the table data.
                      Returns an empty DataFrame if the connection fails or an error occurs.
    """
    conn = get_db_connection()  # Get connection
    if conn is None:
        return pd.DataFrame()  # Return empty DataFrame if connection fails

    try:
        cur = conn.cursor()  # Create a cursor object, which allows database interaction
        query = f'SELECT "{ORDER_DATE_COL}", "{PRODUCT_LINE_COL}", "{SALES_COL}", "{ORDER_NUMBER_COL}", "{CUSTOMER_NAME_COL}", "{TERRITORY_COL}", "{COUNTRY_COL}", "{STATE_COL}", "{CITY_COL}", "{STATUS_COL}", "{DEAL_SIZE_COL}", "{PRODUCT_CODE_COL}", "{QUANTITY_ORDERED_COL}", "{PRICE_EACH_COL}", "{MSRP_COL}", "{ORDER_LINE_NUMBER_COL}" FROM {TABLE_NAME}'
        cur.execute(query)  # Get the values via PostgreSQL query
        results = cur.fetchall()  # Retrieve all rows from a database query result
        columns = [desc[0] for desc in cur.description]  # Get the names of columns
        df = pd.DataFrame(results, columns=columns)  # Match the results to column names
        df[ORDER_DATE_COL] = pd.to_datetime(df[ORDER_DATE_COL])  # ORDERDATE becomes a datetime value
        return df

    except psycopg2.Error as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error
    finally:  # logs out of connection
        if conn:
            cur.close()
            conn.close()  # [cite: 7, 8, 9, 10]

# --- Data Filtering ---

def filter_data(data: pd.DataFrame, column: str, values: List[str]) -> pd.DataFrame:
    """
    Filters a DataFrame based on values in a specific column.

    Args:
        data (pd.DataFrame): The DataFrame to filter.
        column (str): The name of the column to filter by.
        values (List[str]): A list of values to include in the filtered DataFrame.

    Returns:
        pd.DataFrame: The filtered DataFrame.
                      Returns the original DataFrame if `values` is empty.
                      Returns an empty DataFrame if the column name or values list is invalid.
    """
    if not values:  # if values List is empty
        return data  # returns the data DataFrame in values
    if not isinstance(column, str):
        st.error("Invalid column name.")
        return pd.DataFrame()  # Return empty dataframe to avoid errors
    if not isinstance(values, list):
        st.error("Invalid values list.")
        return pd.DataFrame()  # Return empty dataframe to avoid errors
    return data[data[column].isin(values)]  # if the values match fit the data frame [cite: 10, 11]

# --- KPI Calculations ---

@st.cache_data
def calculate_kpis_top(data: pd.DataFrame) -> List[float]:
    """
    Calculates key performance indicators (KPIs) related to overall sales performance.

    Args:
        data (pd.DataFrame): The DataFrame containing sales data.

    Returns:
        List[float]: A list containing total sales (formatted as string) and total orders.
    """
    total_sales = data[SALES_COL].sum()
    sales_in_m = f"{total_sales / 1000000:.2f}M"
    total_orders = data[ORDER_NUMBER_COL].nunique()
    return [sales_in_m, total_orders]  # [cite: 11]

def calculate_kpis_cust(data: pd.DataFrame) -> List[float]:
    """
    Calculates KPIs related to customer behavior.

    Args:
        data (pd.DataFrame): The DataFrame containing sales data.

    Returns:
        List[float]: A list containing unique customers, repeat customer rate (formatted as string),
                     and churn rate (formatted as string).
    """

    unique_customers = data[CUSTOMER_NAME_COL].nunique()
    total_orders = data[ORDER_NUMBER_COL].nunique()
    repeat_cust_rate = f"{total_orders / unique_customers:.2f}%" if unique_customers != 0 else 0
    customer_last_order = data.groupby(CUSTOMER_NAME_COL)[ORDER_DATE_COL].max()
    now = data[ORDER_DATE_COL].max() + timedelta(days=1)
    churn_threshold = timedelta(days=90)
    churned_customers = customer_last_order[now - customer_last_order > churn_threshold].index
    total_customers = data[CUSTOMER_NAME_COL].nunique()
    churn_rate = f"{len(churned_customers) / total_customers:.2f}%" if total_customers > 0 else 0
    return [unique_customers, repeat_cust_rate, churn_rate]  # [cite: 11, 12]

def calculate_kpis_prod(data: pd.DataFrame) -> List[float]:
    """
    Calculates KPIs related to product performance.

    Args:
        data (pd.DataFrame): The DataFrame containing sales data.

    Returns:
        List[float]: A list containing quantity sold and inventory turnover (formatted as string).
    """
    quantity_sold = data[QUANTITY_ORDERED_COL].sum()
    cost_of_goods_sold = (data[MSRP_COL] * data[QUANTITY_ORDERED_COL]).sum()
    average_inventory = data[QUANTITY_ORDERED_COL].mean()
    inventory_turnover = f"{cost_of_goods_sold / average_inventory:.2f}" if average_inventory > 0 else 0
    return [quantity_sold, inventory_turnover]  # [cite: 12, 13]

def calculate_kpis_ord(data: pd.DataFrame) -> List[float]:
    """
    Calculates KPIs related to order metrics.

    Args:
        data (pd.DataFrame): The DataFrame containing sales data.

    Returns:
        List[float]: A list containing average order quantity (formatted as string),
                     average order value (formatted as string), and order fulfillment rate (formatted as string).
    """
    total_sales = data[SALES_COL].sum()
    total_orders = data[ORDER_NUMBER_COL].nunique()
    quantity_sold = data[QUANTITY_ORDERED_COL].sum()
    avg_order_val = f"{total_sales / total_orders / 1000:.2f}K" if total_orders != 0 else 0
    avg_order_quant = f"{quantity_sold / total_orders:.2f}" if total_orders != 0 else 0
    order_fufill_rate = f"{(total_orders * 100) / data[ORDER_NUMBER_COL].count():.2f}%" if data[ORDER_NUMBER_COL].count() != 0 else "0.00%"
    return [avg_order_quant, avg_order_val, order_fufill_rate]  # [cite: 13]

def calculate_kpis_sale(data: pd.DataFrame) -> List[float]:
    """
    Calculates KPIs related to sales profitability.

    Args:
        data (pd.DataFrame): The DataFrame containing sales data.

    Returns:
        List[float]: A list containing profit margin (formatted as string).
    """
    data['Profit'] = (data[PRICE_EACH_COL] - data[MSRP_COL]) * data[QUANTITY_ORDERED_COL]
    total_sales = data[SALES_COL].sum()
    total_profit = data['Profit'].sum()
    profit_margin = f"{total_profit / total_sales * 100:.2f}%" if total_sales > 0 else 0
    return [profit_margin]  # [cite: 13, 14]

# --- KPI Display ---

def display_kpi_metrics(kpis: List[float], kpi_names: List[str], descriptions: List[str]):
    """
    Displays KPI metrics in Streamlit metric components.

    Args:
        kpis (List[float]): A list of KPI values.
        kpi_names (List[str]): A list of KPI names corresponding to the values.
        descriptions (List[str]): A list of descriptions for each KPI.
    """
    for i, (col, (kpi_name, kpi_value, description)) in enumerate(zip(st.columns(4), zip(kpi_names, kpis, descriptions))):
        col.metric(label=kpi_name, value=kpi_value, help=description)  # [cite: 14]

# --- RFM Analysis ---

def calculate_rfm(data: pd.DataFrame):
    """
    Calculates RFM (Recency, Frequency, Monetary) values for each customer.

    Args:
        data (pd.DataFrame): The DataFrame containing sales data.

    Returns:
        pd.DataFrame: A DataFrame with RFM values for each customer.
    """

    now = data[ORDER_DATE_COL].max() + timedelta(days=1)
    rfm = data.groupby(CUSTOMER_NAME_COL).agg({
        ORDER_DATE_COL: lambda x: (now - x.max()).days,
        ORDER_NUMBER_COL: 'nunique',
        SALES_COL: 'sum'
    })
    rfm.rename(columns={ORDER_DATE_COL: 'Recency', ORDER_NUMBER_COL: 'Frequency', SALES_COL: 'Monetary'}, inplace=True)
    return rfm  # [cite: 14, 15]

def segment_customers(rfm: pd.DataFrame):
    """
    Segments customers based on their RFM scores.

    Args:
        rfm (pd.DataFrame): The DataFrame containing RFM values.

    Returns:
        pd.DataFrame: The DataFrame with customer segments added.
    """
    st.subheader("Customer Segmentation (RFM)")
    quantiles = rfm.quantile(q=[0.25, 0.5, 0.75])

    def r_score(x, p, d):  # recency, column name, column values
        if x <= d[p][0.25]:  # if x is with 25th quartile
            return 4  # score = 4
        elif x <= d[p][0.50]:  # if x is with 50th quartile
            return 3  # score = 3
        elif x <= d[p][0.75]:  # if x is with 75th quartile
            return 2  # score = 2
        else:
            return 1  # else, score = 1

    def fm_score(x, p, d):  # frequency and monetary, column name, column values
        if x <= d[p][0.25]:  # if x is with 25th quartile
            return 1  # score = 1
        elif x <= d[p][0.50]:  # if x is with 50th quartile
            return 2  # score = 2
        elif x <= d[p][0.75]:  # if x is with 75th quartile
            return 3  # score = 3
        else:
            return 4  # else, #score = 4

    rfm['R_Quartile'] = rfm['Recency'].apply(r_score, args=('Recency', quantiles)).astype(int)  # Recency Quartile
    rfm['F_Quartile'] = rfm['Frequency'].apply(fm_score, args=('Frequency', quantiles)).astype(int)  # Frequency Quartile
    rfm['M_Quartile'] = rfm['Monetary'].apply(fm_score, args=('Monetary', quantiles)).astype(int)  # Monetary Quartile
    rfm['RFM_Segment'] = rfm.R_Quartile.map(str) + rfm.F_Quartile.map(str) + rfm.M_Quartile.map(str)  # RFM Quartile
    rfm['RFM_Score'] = rfm[['R_Quartile', 'F_Quartile', 'M_Quartile']].sum(axis=1)  # RFM Score

    def segment_customer(row):
        if row['RFM_Score'] >= 9:
            return 'Champions'
        elif row['RFM_Score'] >= 6:
            return 'Potential Loyalists'
        elif row['RFM_Score'] >= 5:
            return 'Promising'
        elif row['RFM_Score'] >= 4:
            return 'Need Attention'
        else:
            return 'About To Sleep'
    rfm['Customer_Segment'] = rfm.apply(segment_customer, axis=1)
    return rfm  #

# --- Sales Forecasting ---

def forecast_sales(data: pd.DataFrame, forecast_days: int):
    """
    Forecasts future sales using an ARIMA model.

    Args:
        data (pd.DataFrame): The DataFrame containing sales data, indexed by order date.
        forecast_days (int): The number of days to forecast.

    Returns:
        pd.DataFrame: A DataFrame with forecasted sales, indexed by date.
                      Returns an empty DataFrame if there's no sales data or an error occurs during fitting.
    """

    sales_data = data.groupby(ORDER_DATE_COL)[SALES_COL].sum()

    if sales_data.empty:
        print("Warning: No sales data for forecasting.")
        return pd.DataFrame(columns=['Forecasted Date', 'Forecasted Sales']).set_index('Forecasted Date')

    try:
        model = sm.tsa.ARIMA(sales_data, order=(5, 1, 0))  # You might need to tune the order parameters
        model_fit = model.fit()
        forecast_steps = forecast_days
        forecast = model_fit.forecast(steps=forecast_steps).apply(currency_format)

        last_date = sales_data.index[-1]
        future_dates = [last_date + timedelta(days=i + 1) for i in range(forecast_steps)]
        forecast_df = pd.DataFrame({'Forecasted Date': future_dates, 'Forecasted Sales': forecast})
        forecast_df = forecast_df.set_index('Forecasted Date')
        return forecast_df

    except Exception as e:
        print(f"Error during ARIMA fitting: {e}")
        print("Consider checking data stationarity, model order, or trying a different forecasting method.")
        return pd.DataFrame(columns=['Forecasted Date', 'Forecasted Sales']).set_index('Forecasted Date')  #

# --- Sidebar Filters ---

def display_sidebar(data: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """
    Displays the sidebar with various filtering options.

    Args:
        data (pd.DataFrame): The main DataFrame to filter.

    Returns:
        Tuple: A tuple containing selected filter values:
               (start_date, end_date, selected_product_lines, selected_territories,
                selected_countries, selected_states, selected_cities, selected_statuses,
                selected_dealsize, selected_productcode)
    """

    st.sidebar.header("Filters")
    years = sorted(data[ORDER_DATE_COL].dt.year.unique())
    # Filter quarters based on available years
    quarters = []
    for y in years:
        for q in range(1, 5):
            start_q = datetime(y, (q - 1) * 3 + 1, 1)
            end_q = get_quarter_end(f"Q{q} {y}")
            if data[ORDER_DATE_COL].min() <= end_q and data[ORDER_DATE_COL].max() >= start_q:
                quarters.append(f"Q{q} {y}")

    # Filter months based on available years
    months = []
    for y in years:
        for m in range(1, 13):
            start_m = datetime(y, m, 1)
            end_m = get_month_end(f"{y}-{m:02}")
            if data[ORDER_DATE_COL].min() <= end_m and data[ORDER_DATE_COL].max() >= start_m:
                months.append(f"{y}-{m:02}")

    st.sidebar.subheader("Date Range")
    time_filter = st.sidebar.selectbox("Time Filter", ["Year", "Quarter", "Month"], key='start_time_filter')

    if time_filter == "Quarter":
        start_quarter = st.sidebar.selectbox("Start Quarter", quarters, key='start_quarter')
        start_date = get_quarter_start(start_quarter)
        end_quarter = st.sidebar.selectbox("End Quarter", quarters, key='end_quarter')
        end_date = get_quarter_end(end_quarter)
    elif time_filter == "Month":
        start_month = st.sidebar.selectbox("Start Month", months, key='start_month')
        start_date = get_month_start(start_month)
        end_month = st.sidebar.selectbox("End Month", months, key='end_month')
        end_date = get_month_end(end_month)
    else:
        start_year = st.sidebar.selectbox("Start Year", years, key='start_year')
        start_date = datetime(start_year, 1, 1)
        end_year = st.sidebar.selectbox("End Year", years, key='end_year')
        end_date = datetime(end_year, 12, 31)

    product_lines = sorted(data[PRODUCT_LINE_COL].unique())
    selected_product_lines = []

    st.sidebar.subheader("Products")
    for product_line in product_lines:
        if st.sidebar.checkbox(product_line, value=True):
            selected_product_lines.append(product_line)

    selected_statuses = st.sidebar.multiselect("Select Order Statuses", data[STATUS_COL].unique())
    selected_dealsize = st.sidebar.multiselect("Select Dealsize", data[DEAL_SIZE_COL].unique())
    selected_productcode = st.sidebar.multiselect("Select Product Code", data[PRODUCT_CODE_COL].unique())

    st.sidebar.subheader("Geographic Sections")
    selected_territories = st.sidebar.multiselect("Select Territories", data[TERRITORY_COL].unique())
    filtered_countries = data[data[TERRITORY_COL].isin(selected_territories)][COUNTRY_COL].unique() if selected_territories else data[COUNTRY_COL].unique()
    selected_countries = st.sidebar.multiselect("Select Countries", filtered_countries)

    filtered_states = []
    filtered_cities = []

    if selected_countries:
        filtered_data_country = data[data[COUNTRY_COL].isin(selected_countries)]
        if any(country in selected_countries for country in ['USA', 'Canada', 'Australia', 'UK', 'Japan', 'France', 'Philippines', 'Sweden', 'Singapore', 'Italy', 'Denmark', 'Belgium', 'Germany', 'Switzerland', 'Ireland']):
            filtered_states = filtered_data_country[STATE_COL].unique()
            selected_states = st.sidebar.multiselect("Select States", filtered_states)
            if selected_states:
                filtered_cities = filtered_data_country[filtered_data_country[STATE_COL].isin(selected_states)][CITY_COL].unique()
            else:
                selected_states = None
                filtered_cities = filtered_data_country[CITY_COL].unique()
        else:
            filtered_cities = filtered_data_country[CITY_COL].unique()
        selected_cities = st.sidebar.multiselect("Select Cities", filtered_cities)
    else:
        selected_states = []
        selected_cities = []
    return start_date, end_date, selected_product_lines, selected_territories, selected_countries, selected_states, selected_cities, selected_statuses, selected_dealsize, selected_productcode  #

# --- Date Range Functions ---

def get_quarter_start(quarter):
    """
    Gets the start date of a given quarter.

    Args:
        quarter (str): The quarter in the format "Q1 2023".

    Returns:
        datetime: The first day of the quarter.
    """
    q, y = quarter.split(" ")
    q_num = int(q[1])
    year = int(y)
    if q_num == 1:
        return datetime(year, 1, 1)
    elif q_num == 2:
        return datetime(year, 4, 1)
    elif q_num == 3:
        return datetime(year, 7, 1)
    else:
        return datetime(year, 10, 1)  #

def get_quarter_end(quarter):
    """
    Gets the end date of a given quarter.

    Args:
        quarter (str): The quarter in the format "Q1 2023".

    Returns:
        datetime: The last day of the quarter.
    """

    q, y = quarter.split(" ")
    q_num = int(q[1])
    year = int(y)
    if q_num == 1:
        return datetime(year, 3, 31)
    elif q_num == 2:
        return datetime(year, 6, 30)
    elif q_num == 3:
        return datetime(year, 9, 30)
    else:
        return datetime(year, 12, 31)  #

def get_month_start(month):
    """
    Gets the start date of a given month.

    Args:
        month (str): The month in the format "2023-01".

    Returns:
        datetime: The first day of the month.
    """
    year, month_num = map(int, month.split("-"))
    return datetime(year, month_num, 1)  #

def get_month_end(month):
    """
    Gets the end date of a given month.

    Args:
        month (str): The month in the format "2023-01".

    Returns:
        datetime: The last day of the month.
    """

    year, month_num = map(int, month.split("-"))
    if month_num == 12:
        return datetime(year, month_num, 31)
    else:
        return datetime(year, month_num + 1, 1) - timedelta(days=1)  #

# --- Formatting Function ---

def currency_format(val):
    """
    Formats a numeric value as currency.

    Args:
        val (float or int): The value to format.

    Returns:
        str: The formatted currency string.
    """
    if val < 0:
        return f'-${abs(val):,.2f}'
    else:
        return f'${val:,.2f}'

# --- Chart Display Functions ---

def display_charts_cust(data: pd.DataFrame):
    """
    Displays charts related to customer analysis, including Customer Lifetime Value (CLTV)
    and top customers by sales.

    Args:
        data (pd.DataFrame): The DataFrame containing sales data.
    """

    colcust1, colcust2 = st.columns(2)
    with colcust1:
        st.subheader("Initial Approximation Customer Lifetime Value")
        customer_avg_order_value = data.groupby(CUSTOMER_NAME_COL)[SALES_COL].mean().reset_index()  # Calculate Average Order Value per Customer
        customer_avg_order_value.rename(columns={SALES_COL: 'Average Order Value'}, inplace=True)
        customer_order_counts = data.groupby(CUSTOMER_NAME_COL)[ORDER_NUMBER_COL].nunique().reset_index()  # Calculate Order Frequency per Customer (assuming you have order dates)
        customer_order_counts.rename(columns={ORDER_NUMBER_COL: 'Order Frequency'}, inplace=True)
        customer_lifespan = data.groupby(CUSTOMER_NAME_COL)[ORDER_DATE_COL].agg(['min', 'max']).reset_index()  # Determine Customer Lifespan (based on order dates)
        customer_lifespan['Lifespan (Days)'] = (customer_lifespan['max'] - customer_lifespan['min']).dt.days
        customer_lifespan = customer_lifespan[['CUSTOMERNAME', 'Lifespan (Days)']]
        CLTV = pd.merge(customer_avg_order_value, customer_order_counts, on='CUSTOMERNAME')  # Merge the dataframes
        CLTV = pd.merge(CLTV, customer_lifespan, on='CUSTOMERNAME')
        time_period_days = (data[ORDER_DATE_COL].max() - data[ORDER_DATE_COL].min()).days  # Define a time period for frequency calculation (e.g., 365 days for annual frequency)
        if time_period_days == 0:
            CLTV['Purchase Frequency (Annual)'] = CLTV['Order Frequency']  # Avoid division by zero
        else:
            CLTV['Purchase Frequency (Annual)'] = (CLTV['Order Frequency'] / (time_period_days / 365))
        CLTV['Customer Value (Annual)'] = CLTV['Average Order Value'] * CLTV['Purchase Frequency (Annual)']  # Calculate Customer Value (Annual)
        CLTV['Average Order Value'] = CLTV['Average Order Value'].apply(currency_format)
        CLTV['Initial CLTV'] = CLTV['Customer Value (Annual)'] * (CLTV['Lifespan (Days)'] / 365)  # Calculate Initial CLTV (using the observed lifespan)
        CLTV['Customer Value (Annual)'] = CLTV['Customer Value (Annual)'].apply(currency_format)
        CLTV.sort_values('Initial CLTV', ascending=False, inplace=True)
        CLTV['Initial CLTV'] = CLTV['Initial CLTV'].apply(currency_format)  # Apply currency format
        CLTV = CLTV.set_index('CUSTOMERNAME')
        st.write(CLTV)
    with colcust2:
        st.subheader("Top Customers")
        top_customers = data.groupby(CUSTOMER_NAME_COL)[SALES_COL].sum().reset_index().sort_values('SALES', ascending=False).head(10)
        top_customers['SALES'] = top_customers['SALES'].apply(currency_format)
        top_customers = top_customers.set_index('CUSTOMERNAME')
        st.write(top_customers)  #

def display_charts_prod(data: pd.DataFrame):
    """
    Displays charts related to product performance, including top-performing products by sales
    and total sales by product line.

    Args:
        data (pd.DataFrame): The DataFrame containing sales data.
    """

    colprod1, colprod2 = st.columns(2)
    with colprod1:
        st.subheader("Top-Performing Products by Sales")
        top_products = data.groupby([PRODUCT_CODE_COL, PRODUCT_LINE_COL])[SALES_COL].sum().reset_index().sort_values('SALES', ascending=False).head(10)
        top_products = top_products.set_index('PRODUCTCODE')
        top_products['SALES'] = top_products['SALES'].apply(currency_format)
        st.write(top_products)
    with colprod2:
        st.subheader("Total Sales by Product Line")
        total_sales_by_product_line = data.groupby(PRODUCT_LINE_COL)[SALES_COL].sum().reset_index().sort_values('SALES', ascending=False)
        total_sales_by_product_line = total_sales_by_product_line.set_index('PRODUCTLINE')
        total_sales_by_product_line['SALES'] = total_sales_by_product_line['SALES'].apply(currency_format)
        st.write(total_sales_by_product_line)  #

def display_charts_ord(data: pd.DataFrame):
    """
    Displays charts related to order analysis, including price realization vs. MSRP
    and an order status funnel.

    Args:
        data (pd.DataFrame): The DataFrame containing sales data.
    """

    colord1, colord2 = st.columns(2)
    with colord1:
        st.subheader('Price Realization vs MSRP')  # Display Price Realization vs MSRP
        fig_price_vs_msrp = px.scatter(data, x=PRICE_EACH_COL, y=MSRP_COL)
        st.plotly_chart(fig_price_vs_msrp, key="price")
    with colord2:
        st.subheader('Order Status Funnel')  # Create funnel chart based on order count based on Status
        funnel_data = data[STATUS_COL].value_counts().reset_index()
        funnel_data.columns = ['Status', 'Count']
        fig_funnel = px.funnel(funnel_data, x='Count', y='Status')
        st.plotly_chart(fig_funnel)  #

def display_charts_sales(data: pd.DataFrame):
    """
    Displays charts related to sales trends and analysis, including sales growth over time,
    order volume trends, correlation between order size and sales, and sales over time
    segmented by product line, deal size, and order status.

    Args:
        data (pd.DataFrame): The DataFrame containing sales data.
    """

    colsale1, colsale2 = st.columns(2)
    with colsale1:
        st.subheader('Sales Growth Over Time')
        df_sales_growth = data.groupby(ORDER_DATE_COL)[SALES_COL].sum().pct_change().reset_index()
        fig_sales_growth = px.line(df_sales_growth, x=ORDER_DATE_COL, y=SALES_COL)
        st.plotly_chart(fig_sales_growth, key="growth")
        df_order_volume_trends = data.groupby(ORDER_DATE_COL)[ORDER_NUMBER_COL].count().reset_index()
        st.subheader('Order Volume Trends')
        fig_order_volume_trends = px.line(df_order_volume_trends, x=ORDER_DATE_COL, y=ORDER_NUMBER_COL)
        st.plotly_chart(fig_order_volume_trends, key="volume")
    with colsale2:
        st.subheader("Sales Growth over Time")
        sales_growth = data.groupby('ORDERDATE')['SALES'].sum().pct_change().reset_index().set_index('ORDERDATE')

        def color_sales_growth(val):
            color = 'red' if val < 0 else 'green'
            return f'color: {color}'
        styled_sales_growth = sales_growth.style.format({'SALES': '{:.2%}'}).map(color_sales_growth, subset=['SALES'])
        st.write(styled_sales_growth, key="growthtable")
        dealsize_sales = data.groupby(DEAL_SIZE_COL)[SALES_COL].sum().reset_index()
        st.subheader('Correlation between Order Size and Sales')
        figo = px.bar(dealsize_sales, x='DEALSIZE', y="SALES")
        figo.update_layout(margin=dict(l=20, r=20, t=50, b=20))
        figo.update_xaxes(rangemode='tozero', showgrid=False)
        figo.update_yaxes(rangemode='tozero', showgrid=True)
        st.plotly_chart(figo, use_container_width=True, key="correlation")

    combine_product_lines = st.checkbox("Show Product Lines", value=False)
    combine_dealsize_lines = st.checkbox("Show Dealsize Lines", value=False)
    combine_status_lines = st.checkbox("Show Status Lines", value=False)
    if combine_product_lines:
        st.subheader("Sales Over Time (with Product Line)")
        fig = px.area(data, x=ORDER_DATE_COL, y=SALES_COL, color=PRODUCT_LINE_COL, width=900, height=500)
    elif combine_dealsize_lines:
        st.subheader("Sales Over Time (with Deal Size)")
        fig = px.area(data, x=ORDER_DATE_COL, y=SALES_COL, color=DEAL_SIZE_COL, width=900, height=500)
    elif combine_status_lines:
        st.subheader("Sales Over Time (with Order Status)")
        fig = px.area(data, x=ORDER_DATE_COL, y=SALES_COL, color=STATUS_COL, width=900, height=500)
    else:
        st.subheader("Sales Over Time")
        fig = px.area(data, x=ORDER_DATE_COL, y=SALES_COL, width=900, height=500)
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    fig.update_xaxes(rangemode='tozero', showgrid=False)
    fig.update_yaxes(rangemode='tozero', showgrid=True)
    st.plotly_chart(fig, use_container_width=True)  #

def display_charts_geo(data: pd.DataFrame):
    """
    Displays charts related to geographic analysis, including top-performing regions
    and a map of sales by country.

    Args:
        data (pd.DataFrame): The DataFrame containing sales data.
    """

    colgeo1, colgeo2 = st.columns(2)
    with colgeo1:
        st.subheader("Top-Performing Regions")  # Display Top-Performing Regions
        top_performing_regions = data.groupby([TERRITORY_COL, COUNTRY_COL, CITY_COL])[SALES_COL].sum().reset_index().sort_values('SALES', ascending=False)
        top_performing_regions = top_performing_regions.set_index([TERRITORY_COL, COUNTRY_COL, CITY_COL])
        top_performing_regions['SALES'] = top_performing_regions['SALES'].apply(currency_format)
        st.write(top_performing_regions)
    with colgeo2:
        st.subheader("Sales By Country")  # Display map fo Sales by Country
        sales_by_country = data.groupby(COUNTRY_COL)[SALES_COL].sum().reset_index()
        fig_map = px.choropleth(
            sales_by_country,
            locations='COUNTRY',
            locationmode='country names',
            color='SALES',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_map)  #

def display_future_sales(data: pd.DataFrame):
    """
    Displays sales forecasting results.

    Args:
        data (pd.DataFrame): The DataFrame containing sales data.
    """

    colfore1, colfore2 = st.columns(2)
    with colfore1:
        # Sales Forecasting
        forecast_period = st.slider("Select Forecast Period (days):", min_value=30, max_value=365, value=90, step=30)  # Example slider
        forecast_df = forecast_sales(data, forecast_period)
        st.subheader(f"Sales Forecasting for the Next {forecast_period} Days")
        st.dataframe(forecast_df)
    with colfore2:
        # Visualization
        st.subheader("Historical Sales and Forecast")
        sales_over_time = data.groupby(ORDER_DATE_COL)[SALES_COL].sum()
        forecast_reset = forecast_df.reset_index()
        sales_reset = sales_over_time.reset_index()
        sales_reset.columns = ['Date', 'Sales']
        forecast_reset.columns = ['Date', 'Sales']
        fig = px.line(sales_reset, x='Date', y='Sales')
        fig.add_scatter(x=forecast_reset['Date'], y=forecast_reset['Sales'], mode='lines', name='Forecasted Sales')
        st.plotly_chart(fig)  #

# --- Main Application ---

def main():
    """
    The main function that runs the Streamlit application.
    It sets up the page configuration, authenticates the user,
    loads data, displays the sidebar filters, and renders the various
    KPIs and charts.
    """

    st.set_page_config(
        page_title="KPI Sales Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown("<style> body {background-color: #f4f4f4; color: #222;}.stSidebar {background-color: #488A99;}.stMetric {background-color: #24bec8; padding: 10px; border-radius: 5px;}.st-dj:{background-color:#aa0000}</style>", unsafe_allow_html=True)

    authenticate()  # Require login

    data = load_data('stattable')  # Load data

    if data.empty:
        st.error("Failed to load data from the database. Please check your connection and table.")
        return

    st.title("Your Interactive KPI Sales Dashboard")

    start_date, end_date, selected_product_lines, selected_territories, selected_countries, selected_states, selected_cities, selected_statuses, selected_dealsize, selected_productcode = display_sidebar(data)
    df = pd.DataFrame(data)
    filtered_data = df[((df[ORDER_DATE_COL]) >= start_date) & ((df[ORDER_DATE_COL]) <= end_date)]  # Filter data based on sidebar parameters
    filtered_data = filter_data(filtered_data, TERRITORY_COL, selected_territories)
    filtered_data = filter_data(filtered_data, COUNTRY_COL, selected_countries)
    filtered_data = filter_data(filtered_data, STATE_COL, selected_states)
    filtered_data = filter_data(filtered_data, CITY_COL, selected_cities)
    filtered_data = filter_data(filtered_data, PRODUCT_LINE_COL, selected_product_lines)
    filtered_data = filter_data(filtered_data, STATUS_COL, selected_statuses)
    filtered_data = filter_data(filtered_data, DEAL_SIZE_COL, selected_dealsize)
    filtered_data = filter_data(filtered_data, PRODUCT_CODE_COL, selected_productcode)

    st.header("Top KPI")
    kpis = calculate_kpis_top(filtered_data)
    kpi_names = ["Total Sales", "Total Orders"]
    kpi_descriptions = ["Total revenue generated.", "Number of orders placed."]
    display_kpi_metrics(kpis, kpi_names, kpi_descriptions)

    st.header("KPI for Customer Insights")
    kpis_cu = calculate_kpis_cust(filtered_data)
    kpi_names_cu = ["Unique Customers", "Repeat Customer Rate", "Churn Rate"]
    kpi_descriptions_cu = ["Number of distinct customers.", "Percentage of repeat customers.", "Percentage of customers who stop doing business with a company over a specific period"]
    display_kpi_metrics(kpis_cu, kpi_names_cu, kpi_descriptions_cu)
    display_charts_cust(filtered_data)
    rfm_df = calculate_rfm(filtered_data)
    segmented_customers = segment_customers(rfm_df)
    st.write(segmented_customers)

    st.header("KPI for Product Performance")
    kpis_pr = calculate_kpis_prod(filtered_data)
    kpi_names_pr = ["Quantity Sold", "Inventory Turnover"]
    kpi_descriptions_pr = ["Number of orders sold.", "Rate that inventory stock is sold, or used, and replaced"]
    display_kpi_metrics(kpis_pr, kpi_names_pr, kpi_descriptions_pr)
    display_charts_prod(filtered_data)

    st.header("KPI for Orders and Pricing")
    kpis_or = calculate_kpis_ord(filtered_data)
    kpi_names_or = ["Average Order Quantity", "Average Order Value", "Order Fulfillment Rate"]
    kpi_descriptions_or = ["Average quantity of products per order.", "Average monetary value per order.", "Percentage of orders being succuesses."]
    display_kpi_metrics(kpis_or, kpi_names_or, kpi_descriptions_or)
    display_charts_ord(filtered_data)

    st.header("KPI for Regional Performance")
    display_charts_geo(filtered_data)

    st.header("KPI for Sales Trends and Analysis")
    kpis_sa = calculate_kpis_sale(filtered_data)
    kpi_names_sa = ["Profit Margin"]
    kpi_descriptions_sa = ["Percentage of profit earned by a company in relation to its revenue"]
    display_kpi_metrics(kpis_sa, kpi_names_sa, kpi_descriptions_sa)
    display_charts_sales(filtered_data)

    st.header("Sales Data Forecasting")
    display_future_sales(filtered_data)

if __name__ == '__main__':
    main()