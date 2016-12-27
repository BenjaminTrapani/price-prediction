import numpy as np
import targets

# returns a flat 1D array of inputs

def fetchPreviousPricesIfAvailable(usefulClosePrices, startIndex, prevToFetch):
    result = []
    for index in range(startIndex - prevToFetch, startIndex):
        if prevToFetch < 0:
            result.append(usefulClosePrices[0])
        else:
            result.append(usefulClosePrices[index])

    return result

def _fetch_raw_input(technicals):
    usefulClosePrices = technicals.getUsefulClosePrices()
    smamap, emamap = technicals.getMovingAverages()
    movingAveragePeriods = technicals.movingAveragePeriods
    volumes = technicals.getVolumes()

    resultInput = []
    for closePriceIndex in range(len(usefulClosePrices)):
        resultInput.append(usefulClosePrices[closePriceIndex])
        resultInput.append(volumes[closePriceIndex])

        prevCloses = fetchPreviousPricesIfAvailable(usefulClosePrices, closePriceIndex, 3)
        resultInput += prevCloses

        for movingAveragePeriod in movingAveragePeriods:
            resultInput.append(smamap[movingAveragePeriod][closePriceIndex])
            resultInput.append(emamap[movingAveragePeriod][closePriceIndex])

    return resultInput

#returns two iterators: input, target
def get_iterators(technicals, simulationParams):
    rawInput = _fetch_raw_input(technicals)

    batch_size = simulationParams.batchSize
    num_steps = simulationParams.technicalsPerPrice

    npInput = np.array(rawInput, dtype=np.float32)
    data_len = len(npInput)
    batch_len = data_len // batch_size

    print "data_len = %d" % (data_len)
    print "batch_size = %d" % (batch_size)
    print "batch_len = %d" % (batch_len)
    print "num_steps = %d" % (num_steps)
    epoch_size = (batch_len - 1) // num_steps
    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    def batchData (rawData):
        tempBatchLen = len(rawData) // batch_size
        data = np.zeros([batch_size, tempBatchLen], dtype=np.float32)
        for i in range(batch_size):
            data[i] = rawData[tempBatchLen * i:tempBatchLen * (i + 1)]
        return data

    inputData = batchData(npInput)

    targetArray = targets.build_targets(num_steps, simulationParams.priceChangeScale, technicals)
    batchedTargets = batchData(targetArray)

    for i in range(epoch_size):
        xBeginIndex = i * num_steps
        xEndIndex = xBeginIndex + num_steps
        input = inputData[:, xBeginIndex:xEndIndex]
        target = batchedTargets[:, i]

        yield (input, target)
