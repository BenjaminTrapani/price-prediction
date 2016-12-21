import tensorflow as tf
import simulation_params
import security_technicals
import stock_lstm
import build_input
import time
from datetime import datetime, timedelta

def main(_):
    beginDate = '2009-04-25'
    endDate = '2016-02-29'
    params = simulation_params.SimulationParams(startDate=beginDate, endDate=endDate,
                                                securityTicker='SPY',
                                                movingAveragePeriods=[20, 50, 100],
                                                batchSize=64,
                                                numSteps=8,
                                                hiddenSize=650,
                                                numLayers=100,
                                                keepProb=0.5,
                                                maxGradNorm=5,
                                                numEpochs=10,
                                                lrDecay=0.8,
                                                learningRate=0.1,
                                                initScale=0.04,
                                                priceChangeScale=15,
                                                numPredictionDays=200)

    technicals = security_technicals.Technicals(params)
    technicals.loadDataInMemory()

    def run_epoch(session, m, inputTechnicals, inputParams, eval_op, verbose=False):
        dataLen = len(inputTechnicals.getUsefulClosePrices()) * inputParams.technicalsPerPrice
        epoch_size = ((dataLen // m.batch_size) - 1) // m.num_steps
        print(epoch_size)
        start_time = time.time()
        costs = 0.0
        iters = 0
        m._initial_state = tf.convert_to_tensor(m._initial_state)
        state = m.initial_state.eval()
        for step, (x, y) in enumerate(build_input.get_iterators(inputTechnicals, inputParams)):
            cost, state, _ = session.run([m.cost, m.final_state, eval_op],
                             {m.input_data: x, m.targets: y, m.initial_state: state})
            costs += cost
            iters += m.num_steps

            print_interval = 20
            if verbose and epoch_size > print_interval \
                    and step % (epoch_size // print_interval) == print_interval:
                print("%.3f mse: %.8f speed: %.0f ips" % (step * 1.0 / epoch_size, costs / iters,
                     iters * m.batch_size / (time.time() - start_time)))
        return costs / (iters if iters > 0 else 1)


    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-params.initScale, params.initScale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = stock_lstm.StockLSTM(is_training=True, simulationParams=params)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mtest = stock_lstm.StockLSTM(is_training=False, simulationParams=params)

        tf.initialize_all_variables().run()

        for epoch in xrange(params.numEpochs):
            lr_decay = params.lrDecay ** max(epoch - params.numEpochs, 0.0)
            print 'lr_decay = ' + str(lr_decay)
            m.assign_lr(session, params.learningRate * lr_decay)
            cur_lr = session.run(m.lr)

            mse = run_epoch(session, m, technicals, params, m.train_op, verbose=True)
            vmse = run_epoch(session, mtest, technicals, params, tf.no_op())
            print("Epoch: %d - learning rate: %.3f - train mse: %.3f - test mse: %.3f" %
                  (epoch, cur_lr, mse, vmse))

        params.startDate = params.endDate
        endDatetime = datetime.strptime(params.startDate, "%Y-%m-%d")
        endDatetime += timedelta(days=params.numPredictionDays)
        params.endDate = endDatetime.strftime("%Y-%m-%d")

        technicals = security_technicals.Technicals(params)
        technicals.loadDataInMemory()

        tmse = run_epoch(session, mtest, technicals, params, tf.no_op())
        print("Test mse: %.3f" % tmse)

    return 0

if __name__ == "__main__":
    tf.app.run()