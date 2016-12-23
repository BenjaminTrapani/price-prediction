import numpy as np

def build_targets(num_steps, scale, technicals):
    usefulClosePrices = technicals.getUsefulClosePrices()
    result = []

    numPositiveBuckets = num_steps/2
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
        chunk[correctedIndex] = num_steps * num_steps * num_steps

        for chunkVal in chunk:
            result.append(chunkVal)

    chunk = np.zeros([num_steps], dtype=np.float32)
    chunk[num_steps/2] = num_steps * num_steps * num_steps

    for chunkVal in chunk:
            result.append(chunkVal)

    return np.array(result, dtype=np.float32)

def pretty_print_targets(targets, num_steps, prev_day_price, scale):
    arrayToPrint = np.zeros([2, num_steps], dtype=np.float32)
    stride = float(prev_day_price) / (float(num_steps) * scale)
    beginVal = prev_day_price - (stride * (float(num_steps)/2))
    index = 0
    for i in range(num_steps):
        arrayToPrint[0][index] = (i * stride + beginVal)
        index += 1
    arrayToPrint[1] = targets
    print arrayToPrint

