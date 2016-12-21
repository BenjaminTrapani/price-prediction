import security_technicals
import numpy as np

# returns a flat 1D array of inputs
def _fetch_raw_input(technicals):
    usefulClosePrices = technicals.getUsefulClosePrices()
    smamap, emamap = technicals.getMovingAverages()
    movingAveragePeriods = technicals.movingAveragePeriods
    volumes = technicals.getVolumes()

    resultInput = []
    for closePriceIndex in range(len(usefulClosePrices)):
        resultInput.append(usefulClosePrices[closePriceIndex])
        resultInput.append(volumes[closePriceIndex])
        for movingAveragePeriod in movingAveragePeriods:
            resultInput.append(smamap[movingAveragePeriod][closePriceIndex])
            resultInput.append(emamap[movingAveragePeriod][closePriceIndex])

    return resultInput

def _fetch_raw_output(num_steps, scale, technicals):
    usefulClosePrices = technicals.getUsefulClosePrices()
    result = np.zeros([num_steps * len(usefulClosePrices)], dtype=np.float32)

    numPositiveBuckets = num_steps/2
    curResultIndex = 0
    for closePriceIndex in range(1, len(usefulClosePrices)):
        lastClosePrice = usefulClosePrices[closePriceIndex-1]
        curClosePrice = usefulClosePrices[closePriceIndex]
        priceChange = curClosePrice - lastClosePrice
        fracChange = priceChange / lastClosePrice
        fracChangeInRange = fracChange * float(numPositiveBuckets)
        adjustedFracChange = fracChangeInRange * scale

        rawCenteredIndex = int(adjustedFracChange)
        correctedIndex = rawCenteredIndex + numPositiveBuckets

        if correctedIndex >= num_steps:
            correctedIndex = num_steps-1
        elif correctedIndex < 0:
            correctedIndex = 0

        chunk = np.zeros([num_steps], dtype=np.float32)
        chunk[correctedIndex] = 1.0

        for chunkVal in chunk:
            result[curResultIndex] = chunkVal
            curResultIndex = curResultIndex + 1

    return result

#returns two iterators: input, target
def get_iterators(technicals, simulationParams):
    rawInput = _fetch_raw_input(technicals)

    batch_size = simulationParams.batchSize
    num_steps = simulationParams.numSteps

    npInput = np.array(rawInput, dtype=np.float32)
    data_len = len(npInput)
    batch_len = data_len // batch_size

    print "data_len = %d" % (data_len)
    print "batch size = %d" % (batch_size)
    print "batch_len = %d" % (batch_len)
    print "num_steps = %d" % (num_steps)
    epoch_size = (batch_len - 1) // num_steps
    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    def batchData (rawData):
        data = np.zeros([batch_size, batch_len], dtype=np.float32)
        for i in range(batch_size):
            data[i] = rawData[batch_len * i:batch_len * (i + 1)]
        return data

    inputData = batchData(npInput)

    targetArray = _fetch_raw_output(num_steps, simulationParams.priceChangeScale, technicals)
    targetData = batchData(targetArray)

    for i in range(epoch_size):
        xBeginIndex = i * num_steps
        xEndIndex = xBeginIndex + num_steps
        input = inputData[:, xBeginIndex:xEndIndex]
        target = targetData[:, xBeginIndex:xEndIndex]
        print 'input = ' + str(input)
        print 'target = ' + str(target)
        yield (input, target)
