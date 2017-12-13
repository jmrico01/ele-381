import csv
import datetime
import numpy as np
import matplotlib.pyplot as plt

CAT_SUSP = 0
CAT_PROB = 1
CAT_CONF = 2

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
                category = CAT_SUSP
            elif row[0] == probableStr:
                category = CAT_PROB
            elif row[0] == confirmedStr:
                category = CAT_CONF

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

    data["total"] = np.zeros((len(data[targetCountries[0]]),))
    for country in targetCountries:
        data["total"] += data[country]

    if (plot):
        for country, infected in data.items():
            plt.plot(infected)
            plt.title(country)
            plt.show()
    
    return [data, len(data[targetCountries[0]]), minDate, maxDate]

if __name__ == "__main__":
    ReadData("ebola_data_db_format.csv",
        ["Sierra Leone"], [CAT_CONF], True, 5, True)