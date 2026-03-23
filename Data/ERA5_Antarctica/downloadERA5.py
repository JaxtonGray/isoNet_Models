import os
import cdsapi

def download_era5_antarctica_daily_statistic(var, year, outputfile):
    dataset = "derived-era5-single-levels-daily-statistics"
    request = {
        "product_type": "reanalysis",
        "var": [
            f"{var}"
        ],
        "year": f"{year}",
        "month": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12"
        ],
        "day": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12",
            "13", "14", "15",
            "16", "17", "18",
            "19", "20", "21",
            "22", "23", "24",
            "25", "26", "27",
            "28", "29", "30",
            "31"
        ],
        "daily_statistic": "daily_mean",
        "time_zone": "utc+00:00",
        "frequency": "3_hourly",
        "area": [-50, -180, -90, 180]
    }
    client = cdsapi.Client()
    client.retrieve(dataset, request, outputfile)

if __name__ == "__main__":
    # First define the that the directory to save the data
    os.makedirs("data_files", exist_ok=True)

    # Define the var and years to download
    variables = ["2m_temperature", "total_precipitation"]
    # Then download the data for the specified years and vars
    for var in variables:
        for year in range(1979, 1980):
            download_era5_antarctica_daily_statistic(var, year, 
                                                     os.path.join("data_files", f"era5_antarctica_daily_statistic_{var}_{year}.nc"))