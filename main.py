import torch
import numpy as np

from src.metric import Ct_index
from src.OSTransformer import SurvivalWrapper
from src.labels_encoding import label_to_survival, survival_prediction
from src.losses import SurvivalLogLikelihoodLoss, SurvivalRankingLoss


def main():
    ### DATA
    n_samples = 100
    n_features = 37
    data = np.random.rand(n_samples, n_features)
    data[np.random.choice((0, 1), (n_samples, n_features), p=(0.8, 0.2)) == 1] = np.nan  # Introduce missing values

    ### LABELS
    events = ("censored", "uncensored")
    num_events = len(events) - 1  # The first event is the censored one, so we do not consider it
    max_time = 72  # Maximum time to consider for the survival analysis
    max_survival = 100  # Use this to generate labels, but then the analysis will consider only the time to max_time, setting those patients who survived longer than max_time to "censored"
    labels = np.hstack( ( np.random.choice( events, (n_samples, 1)), np.random.rand(n_samples, 1)*max_survival ), dtype=object )

    survival_label_function = np.vectorize(lambda label: label_to_survival( label, events, max_time ), signature="(n)->(m)")
    survival_labels = survival_label_function(labels)

    ### MODEL
    ## OSTransformer (shared net)
    emb_dim = n_features + 1
    n_heads = emb_dim // 2
    shared_net_params = dict(emb_dim=emb_dim, num_heads=n_heads, output_size=emb_dim)

    ## CustomMLP (CS subnets)
    hidden_sizes = [400, 200]
    cs_subnet_params = dict(hidden_sizes=hidden_sizes)

    model = SurvivalWrapper(num_events=num_events, max_time=max_time, shared_net_params=shared_net_params, cs_subnets_params=cs_subnet_params)

    ### OUTPUTS
    ## Forward pass
    outputs = model(torch.from_numpy(data).float())

    ## Predictions
    predictions = survival_prediction(outputs, num_events, max_time)

    ### Survival Losses
    loss = 0
    criterion1 = SurvivalLogLikelihoodLoss(num_events=num_events, max_time=max_time)
    loss += criterion1(outputs, torch.from_numpy(survival_labels).float().unsqueeze(dim=1))

    criterion2 = SurvivalRankingLoss(num_events=num_events, max_time=max_time)
    loss += criterion2(outputs, torch.from_numpy(survival_labels).float().unsqueeze(dim=1))

    ### Performance
    outputs = torch.cumsum(outputs.view(-1, num_events, max_time), dim=-1).view(-1, num_events*max_time)  # Compute the cumulative incidence function (CIF) cumulative summing the output probabilities
    performance = Ct_index(survival_labels, outputs.detach().numpy(), num_events, max_time)


if __name__ == "__main__":
    main()
