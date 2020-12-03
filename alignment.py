from nngeometry.object.fspace import FMatDense
from nngeometry.object.vector import FVector
from nngeometry.object import PMatImplicit

from nngeometry.generator import Jacobian
from nngeometry.layercollection import LayerCollection
import torch
from torch.nn.functional import one_hot

def alignment(model, output_fn, loader, n_output, centering=True):
    lc = LayerCollection.from_model(model)
    generator = Jacobian(layer_collection=lc,
                         model=model,
                         loader=loader,
                         function=output_fn,
                         n_output=n_output,
                         centering=centering)
    targets = torch.cat([args[1] for args in iter(loader)])
    targets = one_hot(targets).float()
    targets -= targets.mean(dim=0)
    targets = FVector(vector_repr=targets.t().contiguous())

    K_dense = FMatDense(generator)
    yTKy = K_dense.vTMv(targets)
    frobK = K_dense.frobenius_norm()

    align = yTKy / (frobK * torch.norm(targets.get_flat_representation())**2)

    return align.item(), K_dense.get_dense_tensor()

def layer_alignment(model, output_fn, loader, n_output, centering=True):
    lc = LayerCollection.from_model(model)
    alignments = []

    targets = torch.cat([args[1] for args in iter(loader)])
    targets = one_hot(targets).float()
    targets -= targets.mean(dim=0)
    targets = FVector(vector_repr=targets.t().contiguous())

    for l in lc.layers.items():
        # print(l)
        lc_this = LayerCollection()
        lc_this.add_layer(*l)

        generator = Jacobian(layer_collection=lc_this,
                             model=model,
                             loader=loader,
                             function=output_fn,
                             n_output=n_output,
                             centering=centering)

        K_dense = FMatDense(generator)
        yTKy = K_dense.vTMv(targets)
        frobK = K_dense.frobenius_norm()

        align = yTKy / (frobK * torch.norm(targets.get_flat_representation())**2)

        alignments.append(align.item())

    return alignments

def compute_trK(align_dl, model, output_fn, n_output):
    generator = Jacobian(model, align_dl, output_fn, n_output=n_output)
    F = PMatImplicit(generator)
    return F.trace().item() * len(align_dl)
