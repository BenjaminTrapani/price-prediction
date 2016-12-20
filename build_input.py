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

def _fetch_raw_output(batch_size, batch_len, technicals):
    usefulClosePrices = technicals.getUsefulClosePrices()
    result = np.zeros([batch_size, batch_len], dtype=np.float32)

    #range: -100% -- 100%, clamped on upper boundary
    numPositiveBuckets = batch_len/2
    for closePriceIndex in range(1, len(usefulClosePrices)):
        lastClosePrice = usefulClosePrices[closePriceIndex-1]
        curClosePrice = usefulClosePrices[closePriceIndex]
        priceChange = curClosePrice - lastClosePrice
        fracChange = priceChange / lastClosePrice
        rawCenteredIndex = int(fracChange * numPositiveBuckets)
        correctedIndex = rawCenteredIndex + numPositiveBuckets #implies batch_size should always be divisible by 2

        if correctedIndex >= batch_len:
            correctedIndex = batch_len-1

        curColum = np.zeros([batch_len], dtype=np.float32)
        curColum[correctedIndex] = 1.0
        result[correctedIndex] = curColum

    return result

#returns two iterators: input, target
def get_iterators(technicals, simulationParams):
    rawInput = _fetch_raw_input(technicals)

    batch_size = simulationParams.batchSize
    batch_size = batch_size * simulationParams.technicalsPerPrice
    num_steps = simulationParams.numSteps

    npInput = np.array(rawInput, dtype=np.float32)
    data_len = len(npInput)
    batch_len = data_len // batch_size

    print "batch size = %d" % (batch_size)
    print "batch_len = %d" % (batch_len)
    print "num_steps = %d" % (num_steps)
    epoch_size = (batch_len - 1) // num_steps
    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    inputData = np.zeros([batch_size, batch_len], dtype=np.float32)
    for i in range(batch_size):
        inputData[i] = npInput[batch_len * i:batch_len * (i + 1)]

    targetData = _fetch_raw_output(batch_size, batch_len, technicals)

    for i in range(epoch_size):
        xBeginIndex = i * num_steps
        xEndIndex = xBeginIndex + num_steps
        input = inputData[:, xBeginIndex:xEndIndex]
        target = targetData[:, xBeginIndex:xEndIndex]
        print 'input = ' + str(input)
        print 'target = ' + str(target)
        yield (input, target)
