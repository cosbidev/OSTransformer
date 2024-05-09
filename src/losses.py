import torch

__all__ = ["SurvivalLogLikelihoodLoss", "SurvivalRankingLoss"]

class SurvivalLogLikelihoodLoss(torch.nn.Module):
    def __init__(self, num_events: int, max_time: int, eps: float = 1e-08):
        super(SurvivalLogLikelihoodLoss, self).__init__()
        self.num_events = num_events
        self.max_time = max_time
        self.eps = eps

    def get_uncensored_mask(self, labels):
        batch_dim = labels.shape[0]

        mask = torch.zeros((batch_dim, self.num_events, self.max_time)).to(labels.device)

        mask[torch.arange(batch_dim), torch.clamp(labels[:, 0, 0] - 1, min=0).long(), labels[:, 0, 1].long()] = 1
        mask[labels[:, 0, 0] == 0, 0, : ] = 0

        return mask

    def get_censored_mask(self, labels):
        batch_dim = labels.shape[0]

        mask = torch.zeros((batch_dim, self.max_time)).to(labels.device)

        mask[torch.arange(batch_dim), labels[:, 0, 1].long()] = 1

        mask[labels[:, 0, 0] != 0, :] = 0

        return mask

    def forward(self, outputs, labels ):
        outputs = outputs.view(-1, self.num_events, self.max_time)
        uncensored_mask = self.get_uncensored_mask(labels).to(outputs.device)
        censored_mask = self.get_censored_mask(labels).to(outputs.device)

        CIF = torch.cumsum(outputs, dim=-1)
        censored_values = 1 - torch.sum(CIF, dim=1)

        uncensored_map = torch.sign(labels[:, :, 0])

        tmp1 = torch.nansum(torch.sum(uncensored_mask * outputs, dim=2), dim=1, keepdim=True)
        tmp1 = torch.mul( torch.log(tmp1 + self.eps), uncensored_map )

        tmp2 = torch.nansum(censored_mask * censored_values, dim=1, keepdim=True)
        tmp2 = torch.mul( torch.log(tmp2 + self.eps), (1. - uncensored_map) )

        L1 = tmp1 + tmp2
        loss = - torch.nansum(L1)
        return loss


class SurvivalRankingLoss(torch.nn.Module):
    def __init__(self, num_events: int, max_time: int, sigma: float = 0.1):
        super(SurvivalRankingLoss, self).__init__()
        self.num_events = num_events
        self.max_time = max_time
        self.sigma = sigma

    def get_mask(self, labels):
        batch_dim = labels.shape[0]

        tmp1 = torch.repeat_interleave(labels[:, :, 1], batch_dim, dim=1)
        tmp1 = tmp1 < tmp1.transpose(1, 0)

        tmp2 = torch.repeat_interleave(labels[:, :, 0], batch_dim, dim=1)

        tmp2 = tmp2 != 0
        tmp2[labels[:, 0, 0] == 0, :] = 0

        mask = torch.reshape(torch.unsqueeze(tmp1 * tmp2, dim=2), (1, batch_dim, batch_dim))
        return mask

    def forward(self, outputs, labels ):
        outputs = outputs.view(-1, self.num_events, self.max_time)
        batch_dim = outputs.shape[0]

        CIF = torch.cumsum(outputs, dim=-1)

        sample_idx = torch.unsqueeze(torch.arange(batch_dim), dim=1).to(outputs.device)
        k_event_idx = torch.clamp(labels[:, :, 0] - 1, min=0).long()
        k_time_idx = labels[:, :, 1].long()

        tmp1_idx = torch.cat( [sample_idx, k_event_idx, k_time_idx], dim=1 )
        tmp1 = CIF[tmp1_idx.chunk(chunks=3, dim=1)]
        tmp1 = torch.repeat_interleave(torch.unsqueeze(tmp1, dim=2), batch_dim, dim=2)
        tmp1 = torch.transpose(tmp1, 1, 0)

        CIF_ref = torch.transpose(CIF, 2, 0)
        tmp2_idx = torch.cat([k_time_idx, k_event_idx], dim=1)
        tmp2 = CIF_ref[tmp2_idx.chunk(chunks=2, dim=1)]
        tmp2 = torch.transpose(tmp2, 1, 0)

        tmp_num = tmp1 - tmp2

        tmp = torch.exp(- tmp_num/self.sigma )
        mask = self.get_mask(labels).to(outputs.device)

        L2 = torch.nansum(mask*tmp, dim=2).squeeze()

        loss = torch.nansum(L2)

        return loss


if __name__ == "__main__":
    pass
