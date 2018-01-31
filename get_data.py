import csv
import datetime
import numpy as np

CAT_SUSPECTED = 0
CAT_PROBABLE = 1
CAT_CONFIRMED = 2

COUNTRY_TOTAL = "Total"
FATALITY_RATE = 0.71

T = 10000
startData = {
    "Sierra Leone": [
        7e6,
        200,
        0
    ],
    "Guinea": [
        11.8e6,
        250,
        0
    ],
    "Liberia": [
        4.39e6,
        100,
        0
    ],
    COUNTRY_TOTAL: [
        7e6 + 11.8e6 + 4.39e6,
        200 + 250 + 100,
        0
    ]
}
bestParamsSingle = {
    "Sierra Leone": [
        [
            3.41131913825e-09,
            0.0244659446576
        ],
        [
            3.41131913825e-09,
            0.0244659446576,
            FATALITY_RATE
        ],
    ],
    "Guinea": [
        [
            4.48707220505e-10,
            0.00573472410123
        ],
        [
            4.48707220505e-10,
            0.00573472410123,
            FATALITY_RATE
        ],
    ],
    "Liberia": [
        [
            5.78252080412e-09,
            0.0262566408751
        ],
        [
            5.76711809134e-09,
            0.0261867019139,
            FATALITY_RATE
        ],
    ],
    COUNTRY_TOTAL: [
        [
            6.61558851549e-10,
            0.0159484101774
        ],
        [
            6.61558851549e-10,
            0.0159484101774,
            FATALITY_RATE
        ],
    ],
}
bestParamsSpatial = [
    [
        [ 3.41080561109e-09, 3.64413887624e-13, 3.81147081224e-14 ],
        [ 9.28208363787e-15, 4.48354726597e-10, 7.53251380389e-15 ],
        [ 3.03716108699e-12, 6.20850890573e-12, 5.76643991619e-09 ],
    ],
    [ 0.02446594, 0.00573841529276, 0.0261867 ]
]

# Takes a date string, returns a datetime.date object
# Returns None if invalid date string
def ParseDate(dateStr):
    dateStrArray = dateStr.split("-")
    if len(dateStrArray) != 3:
        return None

    year = int(dateStrArray[0])
    month = int(dateStrArray[1])
    day = int(dateStrArray[2])
    return datetime.date(year, month, day)

def ReadData(dataFilePath, targetCountries, categories,
    doSmooth, smoothWnd, plot = False):
    # Returns [data, n, start date, end date]
    N_TIMESERIES = -1
    data = {
        # "country": [ time series ]
    }

    suspectedStr = "Cumulative number of suspected Ebola cases"
    probableStr = "Cumulative number of probable Ebola cases"
    confirmedStr = "Cumulative number of confirmed Ebola cases"
    dataExt = {
        # sample entry:
        # "country": [
        #   [ time series for suspected ],
        #   [ for probable ],
        #   [ for confirmed ]
        # ]
    }

    minDate = datetime.date.max
    maxDate = datetime.date.min

    with open(dataFilePath, "r") as csvFile:
        reader = csv.reader(csvFile)
        rows = []
        for row in reader:
            date = ParseDate(row[2])
            if date == None:
                continue

            rows.append(row)
            if date < minDate:
                minDate = date
            if date > maxDate:
                maxDate = date
        
        #print("Minimum date: " + str(minDate))
        #print("Maximum date: " + str(maxDate))
        #print("Start date: " + str(minDate))
        #print("End date:   " + str(maxDate))
        #print("Maximum index: " + str((maxDate - minDate).days))
        maxDelta = maxDate - minDate
        N_TIMESERIES = maxDelta.days + 1

        for row in rows:
            category = -1
            if row[0] == suspectedStr:
                category = CAT_SUSPECTED
            elif row[0] == probableStr:
                category = CAT_PROBABLE
            elif row[0] == confirmedStr:
                category = CAT_CONFIRMED

            if category == -1:
                continue

            country = row[1]
            if country not in targetCountries:
                continue

            if country not in dataExt:
                dataExt[country] = [
                    [-1] * N_TIMESERIES,
                    [-1] * N_TIMESERIES,
                    [-1] * N_TIMESERIES
                ]

            date = ParseDate(row[2])
            number = int(float(row[3]))
            dayIndex = date - minDate
            dayIndex = dayIndex.days
            dataExt[country][category][dayIndex] = number

    for country in dataExt:
        for category in range(len(dataExt[country])):
            lastValue = -1
            for i in range(N_TIMESERIES):
                if dataExt[country][category][i] == -1:
                    if lastValue != -1:
                        dataExt[country][category][i] = lastValue
                    else:
                        dataExt[country][category][i] = 0
                else:
                    lastValue = dataExt[country][category][i]
            
            i0 = dataExt[country][category][0]
            for i in range(N_TIMESERIES):
                dataExt[country][category][i] -= i0
    
    for country, countryData in dataExt.items():
        data[country] = np.zeros((N_TIMESERIES,))
        for category in categories:
            data[country] += np.array(countryData[category])

        if doSmooth:
            data[country] = np.convolve(data[country],
                np.ones(smoothWnd,) / smoothWnd, mode="valid")

    data[COUNTRY_TOTAL] = np.zeros((len(data[targetCountries[0]]),))
    for country in targetCountries:
        data[COUNTRY_TOTAL] += data[country]
    
    return [data, len(data[targetCountries[0]]), minDate, maxDate]

if __name__ == "__main__":
    ReadData("ebola_data_db_format.csv",
        ["Sierra Leone"], [CAT_CONFIRMED], True, 5, True)