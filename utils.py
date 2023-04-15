from torch.utils.data.dataloader import DataLoader
from prettytable import PrettyTable

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_classwise_f2_score(cmat):
    classwise_f2_scores = dict()
    for i in range(cmat.shape[0]):
        tn = cmat[i][0][0]
        fn = cmat[i][1][0]
        fp = cmat[i][0][1]
        tp = cmat[i][1][1]
        f2 = tp / (tp + 0.2 * fp + 0.8 * fn)
        classwise_f2_scores['f2_class_' + str(i)] = f2
    return classwise_f2_scores

def get_classwise_f2_score_da(cmat, prefix):
    classwise_f2_scores = dict()
    for i in range(cmat.shape[0]):
        tn = cmat[i][0][0]
        fn = cmat[i][1][0]
        fp = cmat[i][0][1]
        tp = cmat[i][1][1]
        f2 = tp / (tp + 0.2 * fp + 0.8 * fn)
        classwise_f2_scores[prefix + 'f2_class_' + str(i)] = f2
    return classwise_f2_scores


class ForeverDataIterator:
    """A data iterator that will never stop producing data"""
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params