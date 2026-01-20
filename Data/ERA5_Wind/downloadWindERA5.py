import os
import cdsapi

def retrieve_era5_wind_data(years, dir=r"data_files"):
    client = cdsapi.Client()

    dataset = "reanalysis-era5-single-levels"
    request =  {'product_type': ['reanalysis'],
        'year': years,
        'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
        'day': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31'],
        'time': ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'],
        'data_format': 'grib',
        'download_format': 'unarchived',
        'variable': ['100m_u_component_of_wind', '100m_v_component_of_wind']}
    target = f"{dir}/era5_wind_{years[0]}_{years[-1]}.grib"

    client.retrieve(dataset, request, target)

# Function that will be called to get the data in spurts of years
def get_wind_data_in_spurts(min_year, max_year, spurt_size=5):
    # Generate year ranges in spurts
    year_ranges = []
    for start_year in range(min_year, max_year + 1, spurt_size):
        end_year = min(start_year + spurt_size - 1, max_year)
        year_ranges.append(list(range(start_year, end_year + 1)))
    
    # Retrieve data for each spurt
    for years in year_ranges:
        retrieve_era5_wind_data(years)

if __name__ == "__main__":
    # Make sure the data_files directory exists
    os.makedirs("data_files", exist_ok=True)

    # Call the function to get data from 1960 to 2025 in spurts of 5 years
    get_wind_data_in_spurts(1960, 2025, spurt_size=5)