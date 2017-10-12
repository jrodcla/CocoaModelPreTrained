import cocoamodel.configvalues as cv

def calc_steps_details(total_examples):
    cv.EXAMPLES_PER_EPOCH = total_examples
    cv.BATCHES_PER_EPOCH = int(cv.EXAMPLES_PER_EPOCH / cv.BATCH_SIZE)
    cv.DECAY_STEPS = cv.BATCHES_PER_EPOCH * cv.EPOCHS_BEFORE_DECAY
    cv.TOTAL_STEPS = cv.BATCHES_PER_EPOCH * cv.NUM_EPOCHS
