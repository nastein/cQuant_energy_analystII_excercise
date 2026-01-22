"""
main.py — cQuant programming exercise 

Run:
  python main.py

Required packages are in requirements.txt
They can be installed using pip install -r requirements.txt

How to use:
- Put the downloaded data file(s) in ./data/raw
- Fill in the TASK sections once you see the PDF
- All outputs go to ./output (CSVs + PNGs)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns 


# =========
# CONFIG
# =========
RAW_DIR = Path("data/raw")
OUT_DIR = Path("output")
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
SPOT_DIR = OUT_DIR / "formattedSpotHistory"
PROF_DIR = OUT_DIR / "hourlyShapeProfiles"


def ensure_dirs():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SPOT_DIR.mkdir(parents=True, exist_ok=True)
    PROF_DIR.mkdir(parents=True, exist_ok=True)


def main():
    ensure_dirs()

    #======
    #task1
    #======
    files = ['ERCOT_DA_Prices_2016.csv',
         'ERCOT_DA_Prices_2017.csv',
         'ERCOT_DA_Prices_2018.csv',
         'ERCOT_DA_Prices_2019.csv']


    df = pd.concat([pd.read_csv(RAW_DIR/f"{file}") for file in files],ignore_index=True)

    #=====
    #task2
    #=====
    #Convert the date to a datetime object
    df['Date'] = pd.to_datetime(df['Date'])

    #Group by settlement point and monthly average (Use start of the month)
    df_monthly = df.groupby(["SettlementPoint", pd.Grouper(key="Date", freq="MS")])["Price"].mean().reset_index()

    #=====
    #task3
    #=====
    #Create year and month columns
    df_monthly['Year'] = df_monthly["Date"].dt.year
    df_monthly['Month'] = df_monthly["Date"].dt.month

    #create output df where I rename columns
    out3 = (
        df_monthly
        .rename(columns={"Price": "AveragePrice"})
        [["SettlementPoint", "Year", "Month", "AveragePrice"]]
    )

    #output to a csv file
    out3.to_csv(TABLE_DIR/f"AveragePriceByMonth.csv",index=False)

    #=====
    #task4
    #=====
    #Isolate out those rows that are hubs
    df_hubs = df[df["SettlementPoint"].str.upper().str.startswith("HB_", na=False)]
    #get rid of prices that are 0 or negative
    df_hubs = df_hubs[df_hubs["Price"] > 0]
    #Create a year column
    df_hubs["Year"] = df_hubs["Date"].dt.year

    #I am asked to compute the hourly volatility as the std deviation 
    #of in log returns of hourly prices
    #I take this to mean I am to compute for each t: Log(p(t)/p(t-1))
    #this is unitless which is nice and measures relative differences
    df_hubs["Log Return"] = df_hubs.groupby("SettlementPoint")["Price"].transform(lambda s: np.log(s).diff())

    #Drop the NaN rows (should be the first hour only)
    df_hubs.dropna(inplace=True)

    #Compute the volatility
    volatility = df_hubs.groupby(["SettlementPoint", "Year"])["Log Return"].std().reset_index(name="HourlyVolatility")

    #=====
    #task5
    #=====
    #output to a csv file
    volatility.to_csv(TABLE_DIR/f"HourlyVolatilityByYear.csv",index=False)

    #=====
    #task6
    #=====
    #Sort the values of the volatility in descending order, and drop the
    #rows that have a repeated year, these will have lower volatility than 
    #the first one
    max_vol_per_year = volatility.sort_values("HourlyVolatility",ascending=False).drop_duplicates("Year")

    #output to a csv file
    max_vol_per_year.to_csv(TABLE_DIR/f"MaxVolatilityByYear.csv",index=False)

    #=====
    #task7
    #=====
    #the supplemental materials file shows 3 csv files where the
    #variable column is the SettlementPoint
    #the second column is the data (so this should contain the hour as well)
    #the next columns will be the price for each of the 24 hours (X1 .. X24)

    #First make a day column which is just the year-day-month
    df['Day'] = df['Date'].dt.date
    #Now make an hour column
    df['Hour'] = df['Date'].dt.hour

    #I can use a pivot table to do this, my indices will be
    #Settlement point and the day, the columns 
    #will be the hour, and the value of each column is the price
    pivot = df.pivot_table(index=["SettlementPoint", "Day"],columns="Hour",values="Price")

    #Change column names from hours to X1 to X24
    pivot.columns = [f"X{h+1}" for h in pivot.columns]
    #Resent the index so things go from 0 to N
    pivot.reset_index(inplace=True)
    #Rename columns to correct format
    pivot.rename(columns={"Day": "Date"},inplace=True)
    pivot.rename(columns={"SettlementPoint": "Variable"},inplace=True)

    #Now output for each settlementpoint the table
    for sp, g in pivot.groupby("Variable",sort=False):
        filename = f'spot_{sp}.csv'
        g.to_csv(SPOT_DIR/f"{filename}",index=False)

    #==========
    #Bonus - Mean Plots
    #==========
    #I was not given the units of (Price) could be $/Mwh, or just $, so I will not assume one

    df_monthly.set_index(df_monthly['Date'],inplace=True)

    fig, ax = plt.subplots(1,1,figsize=(12,5))
    df_monthly
    df_monthly[df_monthly["SettlementPoint"].str.startswith("HB_", na=False)].groupby("SettlementPoint")["Price"].plot(ax=ax)

    ax.set_ylabel("Price")
    ax.set_xlabel("Date")
    ax.legend()
    ax.tick_params(axis='x', labelrotation=45)

    filename = "SettlementHubAveragePriceByMonth.png"
    plt.savefig(FIG_DIR / f"{filename}", dpi=175, bbox_inches="tight")

    fig, ax = plt.subplots(1,1,figsize=(12,5))
    df_monthly[df_monthly["SettlementPoint"].str.startswith("LZ_", na=False)].groupby("SettlementPoint")["Price"].plot(ax=ax)


    ax.set_ylabel("Price")
    ax.set_xlabel("Date")
    ax.legend()
    ax.tick_params(axis='x', labelrotation=45)

    filename = "LoadZoneAveragePriceByMonth.png"
    plt.savefig(FIG_DIR / f"{filename}", dpi=175, bbox_inches="tight")


    #========
    #Bonus – Volatility Plots
    #========
    #decided to use seaborn as matplotlib make bar charts difficult
    plt.figure()
    sns.barplot(data=volatility,x="Year",y="HourlyVolatility",hue="SettlementPoint")
    plt.tight_layout()
    plt.legend()


    filename="SettlementHubHourlyVolatility.png"
    plt.savefig(FIG_DIR/ f"{filename}", dpi=175, bbox_inches="tight")

    #======
    #Bonus - Hourly shape profile comparison
    #======
    #Add month, day of week, and hour columns to our dataframe
    df["Month"] = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.dayofweek   # Monday=0
    df["Hour"] = df["Date"].dt.hour

    #Now for each settlement point, month, day of week, and hour
    #compute the average price
    hourly_avg = df.groupby(['SettlementPoint', 'Month', 'DayOfWeek', 'Hour'])["Price"].mean().reset_index()

    #Now I want to create a new column called hourlyshape
    #This should basically divide by the average over the 24 hours
    #for a given settlement point, month, day of week
    hourly_avg["HourlyShape"] = hourly_avg["Price"]/hourly_avg.groupby(['SettlementPoint', 'Month', 'DayOfWeek'])["Price"].transform("mean")

    #Since a way to format these tables was not given I'm going to
    #Format them in the same way as the csv files that are fed into
    #the cQuant models, i.e. use a pivot table again!
    pivot2 = hourly_avg.pivot_table(index = ["SettlementPoint", "Month", "DayOfWeek"], columns = "Hour", values = "HourlyShape")

    pivot2.columns = [f"X{h+1}" for h in pivot2.columns]
    pivot2.reset_index(inplace=True)

    for sp, g in pivot2.groupby("SettlementPoint"):
        filename = f"profile_{sp}.csv"
        g.to_csv(PROF_DIR/f"{filename}",index=False)


    print("Done. Outputs saved to:", OUT_DIR.resolve())

    #======
    #Bonus - Open Ended Analysis
    #======
    #I did not get to complete this section.
    #What I wanted to do was do some sort of seasonality analysis
    #i.e. find out seasonal patterns + long term trends + anomalies
    #I was going to use the statsmodel seasonal_decompose


    df_analysis = df.copy()
    #make a dataframe where I have the hourly price and each column is the SP
    df_analysis_sp = df_analysis.groupby(['Date','SettlementPoint'])['Price'].min().unstack('SettlementPoint').sort_index()

    #Drop na because statsmodel doesn't do well with this
    df_analysis_sp.dropna(axis="columns",inplace=True)

    #For each settlement point decompose using an additive model with a 24 hour period
    decompositions = {}
    for sp in df_analysis_sp.columns:
        result = seasonal_decompose(df_analysis_sp[sp],model='add',period=24)
        decompositions[sp] = result

    #I wanted to separate out 24 hour cycles, seasonal cycles, as well as
    #Identify anomolies where extreme weather or political events
    #caused prices to change by large amounts, but I couldn't get things
    #to work quickly enough, not sure if seasonal_decompose is good
    #for things with  multiple periodic cycles?


if __name__ == "__main__":
    main()
