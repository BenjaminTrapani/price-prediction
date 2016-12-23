class SimulationParams:
    def __init__(self, startDate, endDate, securityTicker, movingAveragePeriods, batchSize, hiddenSize,
                 numLayers, keepProb, maxGradNorm, numEpochs, lrDecay, learningRate, initScale, priceChangeScale, numPredictionDays):
        self.startDate = startDate
        self.endDate = endDate
        self.securityTicker = securityTicker
        self.movingAveragePeriods = movingAveragePeriods
        self.batchSize = batchSize
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.keepProb = keepProb
        self.maxGradNorm = maxGradNorm
        self.numEpochs = numEpochs
        self.lrDecay = lrDecay
        self.learningRate = learningRate
        self.initScale = initScale
        self.priceChangeScale = priceChangeScale
        self.numPredictionDays = numPredictionDays
        self.technicalsPerPrice = len(movingAveragePeriods) * 2 + 4 #See build_inputs._fetch_raw_input