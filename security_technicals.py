from yahoo_finance import Share
import moving_average
from datetime import datetime, timedelta

class Technicals:
    def __init__(self, simParams):
        self.simParams = simParams
        self.security = Share(self.simParams.securityTicker)
        self.movingAveragePeriods = simParams.movingAveragePeriods

    def loadDataInMemory(self):
        startDateDatetime = datetime.strptime(self.simParams.startDate, "%Y-%m-%d")
        self.maxMovingAveragePeriod = max(self.movingAveragePeriods)
        adjustedStartDate = startDateDatetime - timedelta(days=self.maxMovingAveragePeriod)
        adjustedStartDateStr = adjustedStartDate.strftime("%Y-%m-%d")
        self.cachedData = self.security.get_historical(adjustedStartDateStr, self.simParams.endDate)

        adjustedCloses = []
        for entry in self.cachedData:
            #prepend since historical data is sorted by date descending
            adjustedCloses.insert(0, float(entry['Adj_Close']))
        self.adjustedCloses = adjustedCloses

        volumes = []
        for entry in self.cachedData:
            volumes.insert(0, float(entry['Volume']))

        self.volumes = volumes

    def getClosePrices(self):
        return self.adjustedCloses

    def getUsefulClosePrices(self):
        return self.adjustedCloses[:self.maxMovingAveragePeriod]

    def getMovingAverages(self):
        resultSMAs = {}
        resultEMAs = {}
        for period in self.movingAveragePeriods:
            tempSMA, tempEMA = moving_average.computeSimpleAndExponentialMovingAverage(self.getClosePrices(), period)
            resultSMAs[period] = tempSMA
            resultEMAs[period] = tempEMA
        return resultSMAs, resultEMAs

    def getVolumes(self):
        return self.volumes


