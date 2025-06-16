'''
This is a MACE re-implementation by Edvin Smajlovic for my bachelor thesis.
It is mainly coded in Pytorch and e3nn.
The folds are required to be in the "folds" directory, and a qm7.mat file in the "data" directory, and a checkpoints directory needs to be available.
To call this script, run:
python mymacepy.py <max_l> <num_channels> <correlation> <interactions> <cutoff>
To for sure disable the Symmetric Contraction, set correlation to anything less than 1.
'''
import torch
torch.serialization.add_safe_globals([slice])
from e3nn import nn, o3
from e3nn.o3 import wigner_D
import math
import numpy as np
import scipy
from ase.io import read, write
import sys, os
import torch.nn.functional as F
from matscipy.neighbours import neighbour_list
from mace.tools.scatter import scatter_sum #this is a torch function, just cut down to avoid c++ and stuff
from mace.modules.wrapper_ops import SymmetricContractionWrapper
torch.set_printoptions(sci_mode=False)
import matplotlib.pyplot as plt

class MyMace(torch.nn.Module):
    def __init__(self, atomtypes, max_l = 1, num_channels=32, rcutoff=3, correlation=3, interactions=2):
        super(MyMace, self).__init__()
        self.register_buffer("atomtypes", torch.tensor(atomtypes, dtype=torch.int64))
        self.max_l = max_l
        self.num_channels = num_channels
        self.rcutoff = float(rcutoff)
        self.correlation = correlation
        self.chemicalEmbedding = torch.nn.Linear(atomtypes, num_channels,bias=False)
        self.W = torch.nn.Linear(num_channels, num_channels, bias=False)
        self.interactions = interactions
        num_bessel = 8
        self.bessel = BesselBasisandPolyCutoff(rcutoff, num_bessel=num_bessel, num_poly=6)
        nodeFeatsIrrep = o3.Irreps(f"{num_channels}x0e")
        edgeAttrsIrrep = o3.Irreps(" + ".join([f"1x{x}e" for x in range(0,max_l+1)]))
        targetIrrep = o3.Irreps(" + ".join([f"{num_channels}x{x}e" for x in range(0,max_l+1)]))
        irreps_middle, instructions = tensorpathsforirrepsmiddle(nodeFeatsIrrep, edgeAttrsIrrep, targetIrrep)
        self.conv_tp = o3.TensorProduct(nodeFeatsIrrep,edgeAttrsIrrep,irreps_middle,instructions=instructions,shared_weights=False, internal_weights=False)
        self.conv_tp_weights = nn.FullyConnectedNet([num_bessel]+[64,64,64]+[num_channels*(max_l+1)],F.silu) #Think the 64s is just... chosen just cause?
        self.linearinteraction = o3.Linear(irreps_middle,targetIrrep, shared_weights=True, internal_weights=True)

        if correlation > 0:
            self.symmetric_contraction = SymmetricContractionWrapper(
                irreps_in=targetIrrep,
                irreps_out=targetIrrep,
                correlation=correlation,
                num_elements=atomtypes,
                cueq_config=None
            )
        self.linearproduct = o3.Linear(targetIrrep, targetIrrep, shared_weights=True, internal_weights=True)

        for i in range(interactions-1):
            self.add_module(f"interaction_block_{i+1}", InteractionBlock(num_channels=num_channels, max_l=max_l, num_bessel=num_bessel, correlation=correlation, num_elements=atomtypes))

        half = num_channels // 2
        self.linearfinal1 = o3.Linear(o3.Irreps(f"{num_channels}x0e"), o3.Irreps(f"{half}x0e"), shared_weights=True, internal_weights=True)
        self.nonlinearfinal = nn.Activation(irreps_in=o3.Irreps(f"{half}x0e"),acts=[F.silu])
        self.linearfinal2 = o3.Linear(o3.Irreps(f"{half}x0e"),o3.Irreps("1x0e"), shared_weights=True, internal_weights=True)

    def forward(self,config, atomic_numbers, rotate=None):
        nodeembeddings = self.chemicalEmbedding(F.one_hot(atomic_numbers, num_classes=self.atomtypes).float())
        config.set_cell(np.eye(3))
        iedges, jedges, dist, vect = neighbour_list("ijdD", config, cutoff=self.rcutoff)
        vect = torch.tensor(vect, dtype=torch.float32, device="cpu")
        edge_attrs = o3.spherical_harmonics([x for x in range(0,self.max_l+1)],vect, True)
        dist = torch.tensor(dist, dtype=torch.float32, device=nodeembeddings.device)
        edge_feats = self.bessel(dist)
        if rotate is not None:
            edge_attrs = rotate(edge_attrs,2)

        edge_attrs = edge_attrs.to(nodeembeddings.device)
        node_feats = self.W(nodeembeddings)
        tpweights = self.conv_tp_weights(edge_feats)
        mji = self.conv_tp(node_feats[iedges],edge_attrs, tpweights)
        message = scatter_sum(mji, torch.tensor(jedges,dtype=torch.int64,device=nodeembeddings.device), dim=0, dim_size=node_feats.shape[0])
        node_feats = self.linearinteraction(message)
        
        if self.correlation > 0:
            node_feats = reshapeo3to3d(node_feats, self.max_l, self.num_channels)
            node_feats = self.symmetric_contraction(node_feats, F.one_hot(atomic_numbers, num_classes=self.atomtypes).float())
        node_feats = self.linearproduct(node_feats)

        for i in range(self.interactions-1):
            interaction_block = getattr(self, f"interaction_block_{i+1}")
            node_feats = interaction_block(node_feats, edge_feats, edge_attrs, iedges, jedges, atomic_numbers)

        node_feats = reshapeo3to3d(node_feats, self.max_l, self.num_channels)
        equivariancecheck = node_feats
        node_feats = node_feats[:, :, 0]
        final = self.linearfinal2(self.nonlinearfinal(self.linearfinal1(node_feats)))
        return torch.sum(final), equivariancecheck

class InteractionBlock(torch.nn.Module):
    '''Interaction block for my MACE re-implementation.'''
    def __init__(self, max_l = 1, num_channels = 32, num_bessel=8, correlation=3, num_elements=None):
        super(InteractionBlock, self).__init__()
        self.max_l = max_l
        self.num_channels = num_channels
        targetIrrep = o3.Irreps(" + ".join([f"{num_channels}x{x}e" for x in range(0,max_l+1)]))
        self.linear_up = o3.Linear(targetIrrep,targetIrrep)
        edgeAttrsIrrep = o3.Irreps(" + ".join([f"1x{x}e" for x in range(0,max_l+1)]))
        irreps_middle, instructions = tensorpathsforirrepsmiddle(targetIrrep, edgeAttrsIrrep, targetIrrep)
        self.correlation = correlation
        num = irreps_middle.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet([num_bessel]+[64,64,64]+[num],F.silu)
        self.conv_tp = o3.TensorProduct(targetIrrep,edgeAttrsIrrep,irreps_middle,instructions=instructions,shared_weights=False, internal_weights=False)
        self.linear = o3.Linear(irreps_middle, targetIrrep)
        if num_elements is None:
            raise Exception("num_elements must be specified for the symmetric contraction")
        self.atomtypes = num_elements
        if correlation > 0:
            self.symmetric_contraction = SymmetricContractionWrapper(
                irreps_in=targetIrrep,
                irreps_out=targetIrrep,
                correlation=correlation,
                num_elements=num_elements,
                cueq_config=None
            )
        self.linearproduct = o3.Linear(targetIrrep, targetIrrep)

    def forward(self, node_feats, edge_feats, edge_attrs, iedges, jedges, atomic_numbers):
        node_feats = self.linear_up(node_feats)
        tpweights = self.conv_tp_weights(edge_feats)
        mji = self.conv_tp(node_feats[iedges], edge_attrs, tpweights)
        message = scatter_sum(mji, torch.tensor(jedges, dtype=torch.int64, device=node_feats.device), dim=0, dim_size=node_feats.shape[0])
        node_feats = self.linear(message)
        if self.correlation > 0:
            node_feats = reshapeo3to3d(node_feats, self.max_l, self.num_channels)
            node_feats = self.symmetric_contraction(node_feats, F.one_hot(atomic_numbers, num_classes=self.atomtypes).float())
        node_feats = self.linearproduct(node_feats)
        return node_feats
    
class BesselBasisandPolyCutoff(torch.nn.Module):
    def __init__(self, rcutoff, num_bessel=8, num_poly=6):
        super().__init__()
        bessel_weights = np.pi / rcutoff * torch.linspace(1,num_bessel, num_bessel, dtype=torch.get_default_dtype())
        self.register_buffer("bessel_weights", bessel_weights)
        self.register_buffer("rcutoff", torch.tensor(rcutoff, dtype=torch.get_default_dtype()))
        self.register_buffer("prefactor",torch.tensor(np.sqrt(2.0 / rcutoff), dtype=torch.get_default_dtype()))
        self.register_buffer("num_poly", torch.tensor(num_poly, dtype=torch.int))

    def forward(self, number: torch.Tensor) -> torch.Tensor:
        number = number.unsqueeze(1)
        numerator = torch.sin(number * self.bessel_weights.unsqueeze(0))
        bessel_out = self.prefactor * numerator / number #formula for bessel basis

        r_over_rcutoff = bessel_out / self.rcutoff
        c0 = (self.num_poly + 1.0) * (self.num_poly + 2.0) / 2.0
        c1 = self.num_poly * (self.num_poly + 2.0)
        c2 = self.num_poly * (self.num_poly + 1.0) / 2
        poly = 1.0 - c0 * torch.pow(r_over_rcutoff, self.num_poly) + c1 * torch.pow(r_over_rcutoff, self.num_poly + 1) - c2 * torch.pow(r_over_rcutoff, self.num_poly + 2)
        return poly * (bessel_out < self.rcutoff)

def reshapeo3to3d(tensor, max_l, num_channels):
    """
    Reshape a 2d tensor with o3 Irreps to a 3D tensor.
    The input tensor is expected to have shape (N, num_channels*(max_l+1)**2).
    The output tensor will have shape (N, num_channels, max_l+1**2).
    """
    scalars = tensor[:,:num_channels].unsqueeze(-1)
    if max_l >= 1:
        vectors = tensor[:,num_channels:num_channels*4].reshape(tensor.shape[0], num_channels, 3)
    if max_l >= 2:
        tensors = tensor[:,num_channels*4:num_channels*(max_l+1)**2].reshape(tensor.shape[0], num_channels, 5)
    if max_l >= 3:
        raise Exception("max_l >= 3 is not supported in this function. If needed it's not too hard to implement")
    
    if max_l == 0:
        return scalars
    elif max_l == 1:
        return torch.cat((scalars, vectors), dim=-1)
    elif max_l == 2:
        return torch.cat((scalars, vectors, tensors), dim=-1)
    
def tensorpathsforirrepsmiddle(irreps1: o3.Irreps, irreps2: o3.Irreps, target_irreps: o3.Irreps):
    #This takes a lot of inspiration from the MACE version, 
    #but the MACE version is also based on the nequip one, and generally this way of doing it just makes sense
    middleirreps = []
    instructions = []
    for i, (channel, l1) in enumerate(irreps1):
        for j, (_, l2) in enumerate(irreps2):
            for l3 in l1*l2:
                if l3 in target_irreps:
                    middleirreps.append((channel, l3))
                    instructions.append((i, j, len(middleirreps)-1, "uvu"))

    middleirreps = o3.Irreps(middleirreps)
    middleirreps, permut, _ = middleirreps.sort()
    newinstructions = []

    for indexirrep1, indexirrep2, indextargetirrep, m in instructions:
        newinstructions.append((indexirrep1,indexirrep2, permut[indextargetirrep], m, True))
    newinstructions = sorted(newinstructions, key=lambda x: x[2])
    return middleirreps, newinstructions

def write_xyz_multi_configs(filename, atom_types_list, coordinates_list, total_energies):
    """
    Writes an XYZ file with multiple configurations.

    Args:
        filename (str): The name of the output XYZ file.
        atom_types_list (list of lists): A list of lists, where each sublist contains the atom types for a single configuration.
        coordinates_list (list of lists): A list of lists, where each sublist contains the coordinates for a single configuration.
        total_energies (list): A list of total energies for each configuration.

    """
    elements = [
        "",  # placeholder for index 0
        "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",  # 1–10
        "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca",  # 11–20
        "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",  # 21–30
        "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr",  # 31–40
        "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",  # 41–50
        "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",  # 51–60
        "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",  # 61–70
        "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",  # 71–80
        "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",  # 81–90
        "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",  # 91–100
        "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",  # 101–110
        "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"              # 111–118
    ]

    with open(filename, 'w') as f:
        for atom_types, coordinates, total_energy in zip(atom_types_list, coordinates_list, total_energies):
            # Find padding index
            padding_index = np.where(atom_types == 0)[0]

            # Remove padding if found
            if padding_index.size > 0:
                atom_types = atom_types[:padding_index[0]]
                coordinates = coordinates[:padding_index[0]]

            num_atoms = len(atom_types)
            f.write(str(num_atoms) + '\n')  # Number of atoms
            f.write('total_energy=' + str(total_energy) + '\n')  # Total energy
            for j in range(num_atoms):
                f.write(elements[int(atom_types[j])] + ' ' + str(coordinates[j][0]) + ' ' + str(coordinates[j][1]) + ' ' + str(coordinates[j][2]) + '\n')

def makesplits():
    print("Creating splits for QM7 dataset")
    datamat = scipy.io.loadmat("data/qm7.mat")
    AtomEnergies = datamat["T"][0]
    AtomCharge = datamat["Z"]
    AtomLoc = datamat["R"]

    for k,testindices in enumerate(datamat["P"]):
        np.random.shuffle(testindices)
        testAtomCharge = AtomCharge[testindices]
        testAtomLoc = AtomLoc[testindices]
        testAtomEnergies = AtomEnergies[testindices]

        trainAtomCharge = np.delete(AtomCharge, testindices, axis=0)
        trainAtomLoc = np.delete(AtomLoc, testindices, axis=0)
        trainAtomEnergies = np.delete(AtomEnergies, testindices, axis=0)

        write_xyz_multi_configs(f"folds/qm7_train_fold_{k+1}.xyz", trainAtomCharge, trainAtomLoc, trainAtomEnergies)
        write_xyz_multi_configs(f"folds/qm7_test_fold_{k+1}.xyz", testAtomCharge, testAtomLoc, testAtomEnergies)

def runbaseline():
    print("Running baseline model for QM7 dataset")
    MAE = []
    RMSE = []
    MSE = []
    for i in range(1,6):
        dbtrain = read(f"folds/qm7_train_fold_{i}.xyz", index=":")
        dbtest = read(f"folds/qm7_test_fold_{i}.xyz", index=":")

        total_energy = 0
        for i in range(len(dbtrain)):
            total_energy += dbtrain[i].info["total_energy"]
        average_total_energy = total_energy / len(dbtrain)
        #print("Average energy of the trainset: ", average_total_energy) 

        All_energies = []
        for i, config in enumerate(dbtest):
            All_energies.append(config.info["total_energy"])
        All_energies = np.array(All_energies)
        All_energies -= average_total_energy

        MAE.append(np.mean(np.abs(All_energies)))
        RMSE.append(np.sqrt(np.mean(All_energies**2)))
        MSE.append(np.mean(All_energies**2))
    print("Mean absolute error of the baseline model: ", np.mean(MAE))
    print("Root mean squared error of the baseline model: ", np.mean(RMSE))
    print("Mean squared error of the baseline model: ", np.mean(MSE))


def trainingloop(testing=False):
    if testing==False:
        print("Starting training loop for QM7 dataset")
    else:
        print("Making model for testing equivariance")
    #Settings
    cuda = True
    max_l = int(sys.argv[1]) #MACE recommends 0 for speed, 1 is recommended, 2 for max accuracy
    num_channels = int(sys.argv[2]) #from MACE paper, 64 for speed, 128 is recommended, 256 for max accuracy
    interactions = int(sys.argv[4]) #MACE paper recommends 2, but can be changed freely
    r_cutoff = float(sys.argv[5]) #For smaller molecules, 3 seemed to make a good amount of edges
    correlation = int(sys.argv[3]) #MACE paper recommends 3, but can be changed freely
    angular_l = 2 #this is not implemented yet. right now it is always = max_l
    learning_rate = 0.001 #learning rate for the optimizer, can be changed freely
    minruns = 25 #minumum number of runs
    runswithoutimprovement = 15 #number of runs without improvement before stopping
    deltaimprov = 0.1 #minimum improvement in validation loss to count as an improvement

    K = 5 #numbers of folds, can't be changed right now
    datamat = scipy.io.loadmat("data/qm7.mat")
    atomtypes = np.unique(datamat["Z"])[1:]
    
    for k in range(K):
        if k != 1:
            continue
        print(f"Starting fold {k+1} of {K}")
        #We set the model parameters
        model = MyMace(len(atomtypes), max_l=max_l, num_channels=num_channels, rcutoff=r_cutoff, correlation=correlation, interactions=interactions)
        criterion2 = torch.nn.L1Loss()
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        if torch.cuda.is_available() and cuda:
            model = model.cuda()
            criterion = criterion.cuda()

        dbthingy = read(f"folds/qm7_train_fold_{k+1}.xyz", index=":")
        dbtrain = dbthingy[:int(len(dbthingy)*0.9)]
        dbvalidation = dbthingy[int(len(dbthingy)*0.9):]
        epoch = 0
        bestvalidation = 1000 #initial validation loss, should be high enough to not be reached

        while True:
            epoch += 1
            for i, config in enumerate(dbtrain):
                atomic_numbers_tensor = torch.tensor(config.get_atomic_numbers(), dtype=torch.int32, device="cuda" if cuda else "cpu")
                atomic_numbers_tensor = torch.searchsorted(torch.tensor(atomtypes,device="cuda" if cuda else "cpu"), atomic_numbers_tensor)

                pred, _ = model.forward(config, atomic_numbers_tensor)
                y = torch.tensor(config.info["total_energy"], dtype=torch.float32, device="cuda" if cuda else "cpu")
                loss = criterion(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if testing==True:
                return model, atomtypes, cuda
                
            validation_losses_l1 = []
            for i, config in enumerate(dbvalidation):
                atomic_numbers_tensor = torch.tensor(config.get_atomic_numbers(), dtype=torch.int32, device="cuda" if cuda else "cpu")
                atomic_numbers_tensor = torch.searchsorted(torch.tensor(atomtypes,device="cuda" if cuda else "cpu"), atomic_numbers_tensor)

                pred,_ = model.forward(config,atomic_numbers_tensor)
                y = torch.tensor(config.info["total_energy"], dtype=torch.float32, device="cuda" if cuda else "cpu")
                loss = criterion2(pred, y)
                validation_losses_l1.append(loss.item())
            print(f"Epoch {epoch}, Validation loss L1: {np.mean(validation_losses_l1)}")

            #We save the model checkpoint if the validation loss is lower than the previous best
            if np.mean(validation_losses_l1) < (bestvalidation - deltaimprov):
                bestvalidation = np.mean(validation_losses_l1)
                torch.save(model.state_dict(), f"checkpoints/qm7_{max_l}_{num_channels}_{correlation}_{k+1}_model.pth")
                epochswithoutimprovement = 0
            else:
                epochswithoutimprovement += 1
            
            #We stop training if the validation loss is not improving for a while
            if epoch >= minruns and epochswithoutimprovement >= runswithoutimprovement:
                break
        
        #We load the best model and test it on the test set
        model.load_state_dict(torch.load(f"checkpoints/qm7_{max_l}_{num_channels}_{correlation}_{k+1}_model.pth",weights_only=True))
        model.eval()
        dbtest = read(f"folds/qm7_test_fold_{k+1}.xyz", index=":")
        TestL1results = []

        for i, config in enumerate(dbtest):
            atomic_numbers_tensor = torch.tensor(config.get_atomic_numbers(), dtype=torch.int32, device="cuda" if cuda else "cpu")
            atomic_numbers_tensor = torch.searchsorted(torch.tensor(atomtypes,device="cuda" if cuda else "cpu"), atomic_numbers_tensor)

            pred, _ = model.forward(config, atomic_numbers_tensor)
            y = torch.tensor(config.info["total_energy"], dtype=torch.float32, device="cuda" if cuda else "cpu")
            loss = criterion2(pred, y)
            TestL1results.append(loss.item())
        print("----------------------------")
        print(f"Fold {k+1}, Test L1: {np.mean(TestL1results)}")

def testequivariance():
    model, atomtypes, cuda = trainingloop(testing=True)
    print("Testing equivariance of the model")
    dbtest = read("folds/qm7_test_fold_1.xyz", index=":")
    differences = []
    for c, config in enumerate(dbtest):
        rot = torch.rand(3)
        Ds = [wigner_D(l, *rot) for l in range(int(sys.argv[1]) + 1)]
        D_block = torch.block_diag(*Ds)
        def rotate(x,size):
            if size == 3:
                return torch.einsum("nci, ij->ncj", x, D_block)
            if size == 2:
                return torch.einsum("ni, ij->nj", x, D_block)


        atomic_numbers_tensor = torch.tensor(config.get_atomic_numbers(), dtype=torch.int64,device="cuda" if cuda else "cpu")
        atomic_numbers_tensor = torch.searchsorted(torch.tensor(atomtypes,device="cuda" if cuda else "cpu"), atomic_numbers_tensor)
        pred, nodefeatures1 = model.forward(config, atomic_numbers_tensor)
        nodefeatures1 = rotate(nodefeatures1.cpu(), 3)   
        pred2, nodefeatures2 = model.forward(config, atomic_numbers_tensor, rotate=rotate)
        print(f"Equivariance of config{c} is {torch.isclose(nodefeatures1.cpu(),nodefeatures2.cpu(),rtol=1e-03, atol=1e-3).all().item()}")

        if cuda:
            differences.append(np.abs(pred.detach().cpu().numpy()-pred2.detach().cpu().numpy()))
        else:
            differences.append(np.abs(pred.detach().numpy()-pred2.detach().numpy()))
    print("Max difference: ", np.max(differences))
    print("Mean difference: ", np.mean(differences))


def plotting():
    '''
    This function is not used currently, and needs to be changed since the epoch lengths are variable now.
    But for the sake of reproducability, this is what it used to look like 
    '''
    raise Exception("Check function comment")
    L1results1 = np.array(L1results1) #50x5
    L1results2 = np.array(L1results2) #100x5

    fig, ax = plt.subplots()
    x = np.arange(1, 50 + 1)
    x2 = np.arange(1, 100 + 1)
    means = np.mean(L1results, axis=1)
    conf_low = means - 1.96 * np.std(L1results, axis=1) / np.sqrt(K)
    conf_high = means + 1.96 * np.std(L1results, axis=1) / np.sqrt(K)
    means2 = np.mean(L1results2, axis=1)
    conf_low2 = means2 - 1.96 * np.std(L1results2, axis=1) / np.sqrt(K)
    conf_high2 = means2 + 1.96 * np.std(L1results2, axis=1) / np.sqrt(K)

    ax.plot(x, means, label='MyMace', color='b', marker='o')
    ax.fill_between(x, conf_low, conf_high, color='b', alpha=.15)
    ax.plot(x2, means2, label='MyMace no SC', color='orange', marker='x')
    ax.fill_between(x2, conf_low2, conf_high2, color='orange', alpha=.15)
    ax.set_ylim(ymin=0,ymax=100)
    ax.set_title('Mean L1 Loss with 95% Confidence Interval from 5-Fold CV')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean L1 Loss')

    #We add the benchmark other papers got, which is 9.9 and 3.5
    ax.axhline(y=9.9, color='r', linestyle='--', label='Rupp et al.')
    ax.axhline(y=3.5, color='g', linestyle='--', label='Montavon et al.')
    ax.legend()
    plt.show()


if __name__ == "__main__":
    print(f"Starting program with max_l={sys.argv[1]}, num_channels={sys.argv[2]}, correlation={sys.argv[3]}, interactions={sys.argv[4]}, cutoff={sys.argv[5]}")
    try:
        if len(os.listdir("folds")) == 0:
            makesplits()
    except:
        print("Folds directory not found or something")
        raise Exception
    #runbaseline()
    #testequivariance()
    trainingloop()