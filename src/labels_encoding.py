import torch
import numpy as np


__all__ = ["label_to_survival", "survival_prediction"]

def label_to_survival( label, classes: tuple, max_time: int ):
    event = label[0]

    time = np.floor( label[1] )
    new_time = min(time, max_time-1)

    if event != classes[0] and new_time < time:
        event = classes[0]

    classes_map = {klass: idx for idx, klass in enumerate(classes)}

    return np.array([ classes_map[ event ], new_time ], dtype=int)


def survival_prediction(outputs, num_events, max_time):
    outputs = outputs.view(-1, num_events, max_time)
    max_probs, times = torch.max(outputs, dim=2)
    _, preds = torch.max(max_probs, dim=1)
    preds = preds.unsqueeze(dim=1)

    event_times = torch.gather(times, 1, preds)
    prediction = torch.cat((preds.add(1), event_times), dim=1)
    return prediction


if __name__ == "__main__":
    pass
