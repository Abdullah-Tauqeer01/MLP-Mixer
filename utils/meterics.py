class RunningAverage:
    def __init__(self):
        self.total = 0
        self.count = 0

    def update(self, value):
        self.total += value
        self.count += 1

    @property
    def average(self):
        if self.count == 0:
            return 0
        return self.total / self.count

    def reset(self):
        self.total = 0
        self.count = 0


def accuracy(predictions, labels):
    correct_count = sum(1 for pred, label in zip(predictions, labels) if pred == label)
    return correct_count / len(labels)
