# KPIDashboardCaProj

This Streamlit application provides an interactive dashboard for visualizing and analyzing sales data. It connects to a PostgreSQL database, performs various calculations, and displays Key Performance Indicators (KPIs), charts, and sales forecasts.  User authentication is included to secure access.

## Features

* **Data Visualization:** Presents sales data through interactive charts and graphs, including sales trends, product performance, and geographic analysis.
* **KPI Calculation:** Calculates essential sales KPIs, such as total sales, order volume, customer metrics (e.g., churn rate, CLTV), product performance (e.g., inventory turnover), and profitability metrics.
* **Data Filtering:** Allows users to filter data by date ranges, product lines, territories, and other relevant dimensions.
* **RFM Analysis:** Performs Recency, Frequency, and Monetary (RFM) analysis to segment customers and identify valuable customer groups.
* **Sales Forecasting:** Implements ARIMA models to forecast future sales trends.
* **User Authentication:** Secures the dashboard with a basic login system.
* **Database Integration:** Connects to a PostgreSQL database to retrieve and analyze sales data.
* **Configuration File:** Uses a `config.ini` file for easy setup of database connection details and table/column names.

## Technologies Used

* Python
* Streamlit
* Pandas
* Psycopg2
* Plotly
* Statsmodels (for forecasting)
* Hashlib (for password hashing)
* Configparser

## Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure the database connection:**

    * Create a `config.ini` file in the project's root directory.
    * Populate the `config.ini` file with your PostgreSQL database connection details.  Here's an example:

        ```ini
        [database]
        host = your_database_host
        port = your_database_port
        dbname = your_database_name
        user = your_database_user
        password = your_database_password

        [table]
        name = your_sales_table_name

        [columns]
        order_date = OrderDate
        product_line = ProductLine
        sales = Sales
        order_number = OrderNumber
        customer_name = CustomerName
        territory = Territory
        country = Country
        state = State
        city = City
        status = Status
        deal_size = DealSize
        product_code = ProductCode
        quantity_ordered = QuantityOrdered
        price_each = PriceEach
        msrp = MSRP
        order_line_number = OrderLineNumber
        ```
    * **Environment Variables (Optional):** You can also set database connection parameters using environment variables. If environment variables are set, they will override the values in `config.ini`.  The environment variable names should be: `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`.

4.  **Run the Streamlit application:**

    ```bash
    streamlit run your_script_name.py  # e.g., streamlit run caproj.py
    ```

5.  **Access the dashboard:** Open your web browser and go to the URL displayed in the terminal (usually `http://localhost:8501`).

## Data Requirements

The application expects your PostgreSQL database table to have columns with names matching those specified in the `config.ini` file.  Here's a summary of the required columns:

* `OrderDate`:  Date of the order (datetime).
* `ProductLine`:  Category of the product.
* `Sales`:  Sales amount (numeric).
* `OrderNumber`:  Unique identifier for the order.
* `CustomerName`:  Name of the customer.
* `Territory`:  Sales territory.
* `Country`:  Country of the sale.
* `State`:  State of the sale.
* `City`:  City of the sale.
* `Status`:  Order status.
* `DealSize`:  Size of the deal.
* `ProductCode`:  Code of the product.
* `QuantityOrdered`:  Quantity of products ordered.
* `PriceEach`:  Price of each product.
* `MSRP`:  Manufacturer's Suggested Retail Price.
* `OrderLineNumber`:  Line number in the order.

## User Authentication

* The application includes a basic user authentication system.
* The default username is "admin" and the default password is "password".
* **Important:** **You should change the default password in the code for security reasons before deploying the application.** The password is hashed using SHA256.  Look for the `USERS` dictionary in the code.

## Customization

* You can customize the dashboard by modifying the code:
    * **KPI Calculations:** Change the functions in the `calculate_kpis_*` sections to calculate different KPIs.
    * **Charts:** Modify the chart types and styling using Plotly.
    * **Filters:** Add or remove filters in the `display_sidebar` function.
    * **Forecasting:** Adjust the ARIMA model parameters in the `forecast_sales` function.
    * **Styling:** Change the appearance of the dashboard using Streamlit's styling options.

## Disclaimer

This is a basic sales dashboard application.  For production environments, you may need to enhance it with:

* More robust user authentication and authorization.
* Error handling and logging.
* Data validation.
* Performance optimization.
* More advanced forecasting models.
