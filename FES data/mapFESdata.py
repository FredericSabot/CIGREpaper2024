import csv
import geopy.distance

"""
This scripts maps the location of the substation from the fes regional breakdown (data at 132kV) to the transmission model used (230/400kV) based on closest geographical distance.
"""

locations_path = 'original/fes2023_regional_breakdown_gsp_info.csv'
GSP_locations = {}
with open(locations_path, newline='', encoding='utf-8-sig') as csv_file:
    reader = csv.reader(csv_file)
    next(reader)  # Skip header
    for row in reader:
        GSP_locations[row[0]] = (float(row[4]), float(row[5]))  # Lattitude and longitude

locations_path = 'Substation_locations.csv'
substation_locations = {}
with open(locations_path, newline='', encoding='utf-8-sig') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        substation_locations[row[0]] = (float(row[1]), float(row[2]))  # Lattitude and longitude
substations = substation_locations.keys()

def aggregate_FES_data(demand_path, output_path):
    P_gross = {substation: 0 for substation in substations}
    Q_net = {substation: 0 for substation in substations}
    storage = {substation: 0 for substation in substations}
    solar = {substation: 0 for substation in substations}
    wind = {substation: 0 for substation in substations}
    hydro = {substation: 0 for substation in substations}
    other = {substation: 0 for substation in substations}
    with open(demand_path, newline='', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skip header
        for row in reader:
            GSP = row[0]
            closest_substation = ''
            min_distance = 999999
            for substation in substations:
                if GSP not in GSP_locations.keys():  # Missing geo data has been manually added for all GSPs in Scotland, so assume remaining ones are on NGET bus
                    closest_substation = 'NGET'
                elif substation_locations[substation][0] < 54:  # Approximate B7 boundary location
                    closest_substation = 'NGET'
                else:
                    distance = geopy.distance.geodesic(GSP_locations[GSP], substation_locations[substation])
                    if distance < min_distance:
                        closest_substation = substation
                        min_distance = distance

            P_gross[closest_substation] += float(row[4])
            Q_net[closest_substation] += float(row[6])
            storage[closest_substation] += float(row[7])
            solar[closest_substation] += float(row[8])
            wind[closest_substation] += float(row[9])
            hydro[closest_substation] += float(row[10])
            other[closest_substation] += float(row[11])

    with open(output_path, 'w', newline='', encoding='utf-8-sig') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Substation', 'P_gross', 'Q_net', 'storage', 'solar', 'wind', 'hydro', 'other'])
        for substation in substations:
            writer.writerow([substation, P_gross[substation], Q_net[substation], storage[substation], solar[substation], wind[substation], hydro[substation], other[substation]])

aggregate_FES_data('original/SummerPM_2021_leading.csv', 'aggregated/SummerPM_2021_leading.csv')
aggregate_FES_data('original/SummerAM_2021_leading.csv', 'aggregated/SummerAM_2021_leading.csv')
aggregate_FES_data('original/Winter_2021_leading.csv', 'aggregated/Winter_2021_leading.csv')

aggregate_FES_data('original/SummerPM_2030_leading.csv', 'aggregated/SummerPM_2030_leading.csv')
aggregate_FES_data('original/SummerAM_2030_leading.csv', 'aggregated/SummerAM_2030_leading.csv')
aggregate_FES_data('original/Winter_2030_leading.csv', 'aggregated/Winter_2030_leading.csv')